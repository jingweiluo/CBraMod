import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Sampler
import torch.distributed as dist
import math

from datasets.pretraining_dataset import PretrainingDataset
from datasets.moabb_dataset import MoabbPretrainingDataset  # 你上面写的 MoabbPretrainingDataset 放在这里
from models.cbramod import CBraMod
from pretrain_trainer import Trainer
import os
from utils.constants import ROOT_DIR, LMDB_DIR_DICT, SEQ_LEN_DICT, CHAN_NAME_DICT, MOABB_DATASET_LIST
from utils.util import get_ch_coord

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _parse_dataset_list(s: str):
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


class _TaggedDataset(Dataset):
    """同时返回data和ds_id的Dataset。"""
    def __init__(self, base_ds: Dataset, ds_id: int):
        self.base_ds = base_ds
        self.ds_id = int(ds_id)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x = self.base_ds[idx]
        return x, self.ds_id


class MultiDatasetBatchSampler(Sampler[list]):
    def __init__(
        self,
        dataset_sizes: list[int],
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        # [DDP CHANGE] add rank/world_size
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset_sizes = [int(x) for x in dataset_sizes]
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        self.offsets = []
        s = 0
        for n in self.dataset_sizes:
            self.offsets.append(s)
            s += n
        self.total_size = s
        self.epoch = 0
        # [DDP CHANGE]
        self.rank = int(rank)
        self.world_size = int(world_size)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []
        for ds_id, n in enumerate(self.dataset_sizes):
            if self.shuffle:
                local_indices = torch.randperm(n, generator=g).tolist()
            else:
                local_indices = list(range(n))

            for i in range(0, n, self.batch_size):
                chunk = local_indices[i:i + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                offset = self.offsets[ds_id]
                all_batches.append([offset + j for j in chunk])

        # batch-level shuffle
        if self.shuffle and len(all_batches) > 0:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]

        # 在打乱之后、分发之前做截断
        if self.world_size > 1:
            total = len(all_batches)
            total_trunc = (total // self.world_size) * self.world_size
            all_batches = all_batches[:total_trunc]

        # shard batches by rank
        for batch_idx, b in enumerate(all_batches):
            if batch_idx % self.world_size == self.rank:
                yield b

    def __len__(self):
        total = 0
        for n in self.dataset_sizes:
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size

        if self.world_size > 1:
            total = (total // self.world_size)  # 每个rank的batch数（在做了截断后）
        return total


def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
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
    # parser.add_argument('--seq_len', type=int, default=30)  # 默认值；真实 seq_len 会按 ds 切换
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--need_mask', type=bool, default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.5)

    parser.add_argument('--pretrain_dataset', type=str, default='TUEG')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights')

    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--shuffle', type=bool, default=True)

    # Channel select and shuffle
    parser.add_argument('--use_channel_subset', action='store_true', help='disabled by default')
    parser.add_argument('--min_keep_ch', type=int, default=12)
    parser.add_argument('--max_keep_ch', type=int, default=0)
    parser.add_argument('--channel_shuffle', type=bool, default=True)

    # loss type
    parser.add_argument('--train_mode', type=str, default='recon') # contrastive, recon, both

    # for codebook usage penalty loss in contrastive loss
    parser.add_argument('--lambda_codebook', type=float, default=0.1)

    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')
    params = parser.parse_args()

    # [DDP CHANGE] init distributed
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print("==============world_size=============: ", world_size)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # [DDP CHANGE] store for later use
    params.local_rank = local_rank
    params.rank = rank
    params.world_size = world_size


    dataset_list = _parse_dataset_list(params.pretrain_dataset)
    if len(dataset_list) == 0:
        raise ValueError("Empty --pretrain_dataset. Use e.g. --pretrain_dataset TUEG or TUEG,CHBMIT")

    os.makedirs(params.foundation_dir, exist_ok=True)
    setup_seed(params.seed)

    # --------- 构造每个数据集的元信息（ch_names/seq_len/ch_coords）---------
    params.dataset_list = dataset_list
    params.num_datasets = len(dataset_list)
    params.ds_ch_names = {}
    params.ds_seq_len = {}
    params.ds_ch_coords = {}

    # --------- ConcatDataset：每条样本带 ds_id ----------
    tagged_datasets = []
    raw_datasets = []
    dataset_sizes = []

    for ds_id, dataset_name in enumerate(dataset_list):
        dataset_dir = os.path.join(ROOT_DIR, 'lmdb', LMDB_DIR_DICT[dataset_name])
        ds = PretrainingDataset(dataset_dir=dataset_dir)

        raw_datasets.append(ds)

        # 2) 元信息：ch_names / seq_len
        ch_names = CHAN_NAME_DICT[dataset_name]

        sample0 = ds[0]
        if isinstance(sample0, tuple):
            sample0 = sample0[0]
        # sample0 shape: (C, P, W)
        seq_len = int(sample0.shape[1])   # P

        params.ds_ch_names[ds_id] = ch_names
        params.ds_seq_len[ds_id] = seq_len
        params.ds_ch_coords[ds_id] = get_ch_coord(dataset_name)  # (C,3)

        # 4) tagged dataset
        tagged = _TaggedDataset(ds, ds_id=ds_id)
        tagged_datasets.append(tagged)
        dataset_sizes.append(len(tagged))

    # 默认先指向 ds0；训练时每个 batch 会切换
    params.ch_names = params.ds_ch_names[0]
    params.seq_len = params.ds_seq_len[0]
    params.ch_coords = params.ds_ch_coords[0]

    train_dataset = tagged_datasets[0] if len(tagged_datasets) == 1 else ConcatDataset(tagged_datasets)

    # --------- 关键：用 MultiDatasetBatchSampler 保证每批单一 ds ----------
    batch_sampler = MultiDatasetBatchSampler(
        dataset_sizes=dataset_sizes,
        batch_size=params.batch_size,
        drop_last=params.drop_last,
        shuffle=params.shuffle,
        seed=params.seed,
        # [DDP CHANGE]
        rank=params.rank,
        world_size=params.world_size,
    )

    # [DDP CHANGE] choose workers per process
    num_workers = 4  # 你可按机器 CPU 调整；不要每进程 32

    data_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        pin_memory=True,
        persistent_workers=True
    )

    import textwrap
    def _print_full_channels(dataset_name: str, ch_names: list[str], width: int = 120):
        """
        完整打印通道名（不省略），自动按 width 换行。
        """
        s = ", ".join(ch_names)
        wrapped = textwrap.fill(
            s,
            width=width,
            subsequent_indent=" " * 6,  # 让换行后的缩进更美观
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(f"Channels ({dataset_name}) [{len(ch_names)}]:")
        print("  " + wrapped)

    if params.rank == 0:
        print("============== Datasets Summary ==============")
        # ---- 表格：不包含 channels，避免太长 ----
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

        # 计算列宽并打印（纯文本表格）
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
        print("Total datasets      :", len(dataset_list))
        print("Total samples       :", total_samples)
        print("Total batches/epoch :", len(batch_sampler))
        print("Total hours :", total_seconds // 3600)
        print("Total channel hours :", total_channel_seconds // 3600)

        print("==============================================\n")

        # ---- 通道：逐个 dataset 完整打印（不省略）----
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

    trainer = Trainer(params, data_loader, model, batch_sampler=batch_sampler)
    trainer.train(resume_path=params.resume)

    # close lmdb（MOABB 不需要 close）
    for ds in raw_datasets:
        if hasattr(ds, "db") and ds.db is not None:
            ds.db.close()

    # [DDP CHANGE]
    if params.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
