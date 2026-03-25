import argparse
import math
import os
import random
import textwrap

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from datasets.pretraining_dataset import PretrainingDataset
from datasets.moabb_dataset import MoabbPretrainingDataset
from models.cbramod import CBraMod

# from pretrain_trainer import Trainer
from pretrain_trainer_backup import Trainer

from utils.constants import ROOT_DIR, LMDB_DIR_DICT, SEQ_LEN_DICT, CHAN_NAME_DICT, MOABB_DATASET_LIST
from utils.util import get_ch_coord


def setup_seed(seed, rank=0, deterministic=True):
    seed = int(seed) + int(rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _parse_dataset_list(s: str):
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


class _TaggedDataset(Dataset):
    """同时返回 data 和 ds_id 的 Dataset。"""
    def __init__(self, base_ds: Dataset, ds_id: int):
        self.base_ds = base_ds
        self.ds_id = int(ds_id)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x = self.base_ds[idx]
        return x, self.ds_id


class MultiDatasetTemperatureBatchSampler(Sampler[list[int]]):
    """
    每个 batch 只来自单一 dataset。
    数据集级别采样概率:
        p_i ∝ n_i ^ alpha

    新增：
    - 支持每次连续输出多个 batch（chunk）来自同一个 dataset
    - 通过 batches_per_dataset_chunk 控制 chunk 长度
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        batch_size: int,
        alpha: float = 0.5,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        total_batches_per_epoch: int | None = None,
        batches_per_dataset_chunk: int = 1,
    ):
        self.dataset_sizes = [int(x) for x in dataset_sizes]
        self.batch_size = int(batch_size)
        self.alpha = float(alpha)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        self.batches_per_dataset_chunk = max(1, int(batches_per_dataset_chunk))

        self.offsets = []
        s = 0
        for n in self.dataset_sizes:
            self.offsets.append(s)
            s += n
        self.total_size = s

        self.native_batches = []
        for n in self.dataset_sizes:
            if self.drop_last:
                self.native_batches.append(n // self.batch_size)
            else:
                self.native_batches.append((n + self.batch_size - 1) // self.batch_size)

        if total_batches_per_epoch is None:
            total_batches_per_epoch = sum(self.native_batches)
        self.total_batches_per_epoch = int(total_batches_per_epoch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _build_local_indices(self, n: int, g: torch.Generator):
        if self.shuffle:
            return torch.randperm(n, generator=g).tolist()
        return list(range(n))

    def _sample_dataset_batch_counts(self, g: torch.Generator):
        sizes = torch.tensor(self.dataset_sizes, dtype=torch.float64)
        weights = sizes.pow(self.alpha)

        if float(weights.sum()) <= 0:
            weights = torch.ones_like(weights)

        probs = weights / weights.sum()

        sampled_ds = torch.multinomial(
            probs,
            num_samples=self.total_batches_per_epoch,
            replacement=True,
            generator=g,
        )

        counts = [0 for _ in self.dataset_sizes]
        for ds_id in sampled_ds.tolist():
            counts[ds_id] += 1
        return counts, probs.tolist()

    def get_sampling_probs(self):
        sizes = torch.tensor(self.dataset_sizes, dtype=torch.float64)
        weights = sizes.pow(self.alpha)
        probs = weights / weights.sum()
        return probs.tolist()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        target_batch_counts, _ = self._sample_dataset_batch_counts(g)

        # 每个 dataset 先生成自己所有目标 batch
        ds_batches = {}
        for ds_id, n in enumerate(self.dataset_sizes):
            target_num_batches = target_batch_counts[ds_id]
            if target_num_batches <= 0:
                ds_batches[ds_id] = []
                continue

            offset = self.offsets[ds_id]
            needed_num_samples = target_num_batches * self.batch_size

            local_indices = []
            while len(local_indices) < needed_num_samples:
                one_pass = self._build_local_indices(n, g)
                local_indices.extend(one_pass)

            local_indices = local_indices[:needed_num_samples]

            batches = []
            for i in range(0, needed_num_samples, self.batch_size):
                chunk = local_indices[i:i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append([offset + j for j in chunk])

            ds_batches[ds_id] = batches

        # 把每个 dataset 的 batch 切成 chunk，每个 chunk 连续输出多个 batch
        all_batch_chunks = []
        for ds_id, batches in ds_batches.items():
            if len(batches) == 0:
                continue

            for i in range(0, len(batches), self.batches_per_dataset_chunk):
                chunk = batches[i:i + self.batches_per_dataset_chunk]
                if len(chunk) == 0:
                    continue
                all_batch_chunks.append(chunk)

        # 在 chunk 级别打乱，而不是 batch 级别
        if self.shuffle and len(all_batch_chunks) > 0:
            perm = torch.randperm(len(all_batch_chunks), generator=g).tolist()
            all_batch_chunks = [all_batch_chunks[i] for i in perm]

        # flatten 回 batch 列表，但保持 chunk 内连续
        all_batches = [b for chunk in all_batch_chunks for b in chunk]

        # DDP 对齐截断
        if self.world_size > 1:
            total = len(all_batches)
            total_trunc = (total // self.world_size) * self.world_size
            all_batches = all_batches[:total_trunc]

        for batch_idx, b in enumerate(all_batches):
            if batch_idx % self.world_size == self.rank:
                yield b

    def __len__(self):
        total = self.total_batches_per_epoch
        if self.world_size > 1:
            total = total // self.world_size
        return total


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR')

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--in_dim', type=int, default=200)
    parser.add_argument('--out_dim', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=200)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)

    parser.add_argument('--need_mask', type=str2bool, default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    parser.add_argument('--pretrain_dataset', type=str, default='TUEG')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights')

    parser.add_argument('--drop_last', type=str2bool, default=True)
    parser.add_argument('--shuffle', type=str2bool, default=True)

    parser.add_argument('--use_channel_subset', action='store_true', help='disabled by default')
    parser.add_argument('--min_keep_ch', type=int, default=12)
    parser.add_argument('--max_keep_ch', type=int, default=0)
    parser.add_argument('--channel_shuffle', type=str2bool, default=True)

    parser.add_argument('--train_mode', type=str, default='recon')  # contrastive, recon, both
    parser.add_argument('--lambda_codebook', type=float, default=0.1)

    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--deterministic', type=str2bool, default=True)

    parser.add_argument(
        '--dataset_sampling',
        type=str,
        default='temperature',
        choices=['size', 'temperature', 'uniform']
    )
    parser.add_argument('--dataset_alpha', type=float, default=0.5)
    parser.add_argument('--total_batches_per_epoch', type=int, default=-1)

    # 新增：连续多少个 batch 来自同一个 dataset
    parser.add_argument('--batches_per_dataset_chunk', type=int, default=100)

    params = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print("==============world_size=============: ", world_size)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    params.local_rank = local_rank
    params.rank = rank
    params.world_size = world_size

    os.makedirs(params.foundation_dir, exist_ok=True)
    setup_seed(params.seed, rank=params.rank, deterministic=params.deterministic)

    dataset_list = _parse_dataset_list(params.pretrain_dataset)
    if len(dataset_list) == 0:
        raise ValueError("Empty --pretrain_dataset. Use e.g. --pretrain_dataset TUEG or TUEG,CHBMIT")

    for dataset_name in dataset_list:
        if dataset_name not in LMDB_DIR_DICT:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available datasets: {list(LMDB_DIR_DICT.keys())}"
            )

    params.dataset_list = dataset_list
    params.num_datasets = len(dataset_list)
    params.ds_ch_names = {}
    params.ds_seq_len = {}
    params.ds_ch_coords = {}

    tagged_datasets = []
    raw_datasets = []
    dataset_sizes = []

    for ds_id, dataset_name in enumerate(dataset_list):
        dataset_dir = os.path.join(ROOT_DIR, 'lmdb', LMDB_DIR_DICT[dataset_name])
        ds = PretrainingDataset(dataset_dir=dataset_dir)

        raw_datasets.append(ds)

        ch_names = CHAN_NAME_DICT[dataset_name]

        sample0 = ds[0]
        if isinstance(sample0, tuple):
            sample0 = sample0[0]
        seq_len = int(sample0.shape[1])  # sample0: (C,P,W)

        params.ds_ch_names[ds_id] = ch_names
        params.ds_seq_len[ds_id] = seq_len
        params.ds_ch_coords[ds_id] = get_ch_coord(dataset_name)

        tagged = _TaggedDataset(ds, ds_id=ds_id)
        tagged_datasets.append(tagged)
        dataset_sizes.append(len(tagged))

    params.ch_names = params.ds_ch_names[0]
    params.seq_len = params.ds_seq_len[0]
    params.ch_coords = params.ds_ch_coords[0]

    train_dataset = tagged_datasets[0] if len(tagged_datasets) == 1 else ConcatDataset(tagged_datasets)

    if params.dataset_sampling == 'size':
        alpha = 1.0
    elif params.dataset_sampling == 'uniform':
        alpha = 0.0
    else:
        alpha = params.dataset_alpha

    if params.total_batches_per_epoch <= 0:
        total_batches_per_epoch = None
    else:
        total_batches_per_epoch = params.total_batches_per_epoch

    batch_sampler = MultiDatasetTemperatureBatchSampler(
        dataset_sizes=dataset_sizes,
        batch_size=params.batch_size,
        alpha=alpha,
        drop_last=params.drop_last,
        shuffle=params.shuffle,
        seed=params.seed,
        rank=params.rank,
        world_size=params.world_size,
        total_batches_per_epoch=total_batches_per_epoch,
        batches_per_dataset_chunk=params.batches_per_dataset_chunk,
    )

    def seed_worker(worker_id):
        worker_seed = params.seed + params.rank * 1000 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    persistent_workers = params.num_workers > 0

    dataloader_kwargs = dict(
        dataset=train_dataset,
        num_workers=params.num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
    )

    if params.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = params.prefetch_factor

    data_loader = DataLoader(**dataloader_kwargs)

    def _print_full_channels(dataset_name: str, ch_names: list[str], width: int = 120):
        s = ", ".join(ch_names)
        wrapped = textwrap.fill(
            s,
            width=width,
            subsequent_indent=" " * 6,
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(f"Channels ({dataset_name}) [{len(ch_names)}]:")
        print("  " + wrapped)

    if params.rank == 0:
        print("============== Datasets Summary ==============")
        headers = ["ds_id", "dataset", "samples", "n_channels", "seq_len(P)", "hours", "chan_hours"]
        rows = []

        total_samples = 0
        total_seconds = 0
        total_channel_seconds = 0

        for ds_id, dataset_name in enumerate(dataset_list):
            n_samples = dataset_sizes[ds_id]
            ch_names = params.ds_ch_names[ds_id]
            n_ch = len(ch_names)
            seq_len = int(params.ds_seq_len[ds_id])
            hours = n_samples * seq_len // 3600
            chan_hours = n_samples * seq_len * n_ch // 3600

            total_samples += n_samples
            total_seconds += n_samples * seq_len
            total_channel_seconds += n_samples * seq_len * n_ch
            rows.append([ds_id, dataset_name, n_samples, n_ch, seq_len, hours, chan_hours])

        col_w = [len(h) for h in headers]
        for r in rows:
            for i, v in enumerate(r):
                col_w[i] = max(col_w[i], len(str(v)))

        def _fmt_row(r):
            return " | ".join(str(v).ljust(col_w[i]) for i, v in enumerate(r))

        print(_fmt_row(headers))
        print("-+-".join("-" * w for w in col_w))
        for r in rows:
            print(_fmt_row(r))

        print("-------------- Loader/Training ---------------")
        print("Total datasets            :", len(dataset_list))
        print("Total samples             :", total_samples)
        print("Total batches/epoch       :", len(batch_sampler))
        print("Total hours               :", total_seconds // 3600)
        print("Total channel hours       :", total_channel_seconds // 3600)
        print("Sampling strategy         :", params.dataset_sampling)
        print("Sampling alpha            :", alpha)
        print("Batches per dataset chunk :", params.batches_per_dataset_chunk)
        print("Num workers / process     :", params.num_workers)
        print("Persistent workers        :", persistent_workers)

        probs = batch_sampler.get_sampling_probs()
        print("----------- Sampling Probabilities -----------")
        for ds_id, dataset_name in enumerate(dataset_list):
            print(
                f"{dataset_name:<20} "
                f"size={dataset_sizes[ds_id]:<10d} "
                f"prob={probs[ds_id]:.6f}"
            )

        print("==============================================\n")

        print("============== Channels (Full) ===============")
        for ds_id, dataset_name in enumerate(dataset_list):
            ch_names = params.ds_ch_names[ds_id]
            _print_full_channels(dataset_name, ch_names, width=120)
            print("----------------------------------------------")
        print("==============================================")

    model = CBraMod(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward,
        params.seq_len, params.n_layer, params.nhead,
    )

    if params.world_size > 1:
        dist.barrier()

    try:
        trainer = Trainer(params, data_loader, model, batch_sampler=batch_sampler)
        trainer.train(resume_path=params.resume)
    finally:
        for ds in raw_datasets:
            if hasattr(ds, "db") and ds.db is not None:
                ds.db.close()

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()