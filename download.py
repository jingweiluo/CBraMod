import os
import argparse
import gc
import pickle
from typing import Tuple, List

import numpy as np
import lmdb
from utils.constants import SUB_NUM_DICT, SEQ_LEN_DICT
from tqdm import tqdm

from moabb.paradigms import LeftRightImagery, MotorImagery, P300, SSVEP
from moabb.datasets import (
    BNCI2014_002,
    BNCI2014_004,
    BNCI2015_001,
    Kalunga2016,
    Nakanishi2015,
    BNCI2014_008,
    Weibo2014,
    AlexMI,
    Zhou2016,
    Cho2017,
    BI2015a,
    Lee2019_SSVEP,
    Lee2019_ERP,
    Lee2019_MI,
)

# -------------------------
# Registry
# -------------------------
DATA_DICT = {
    "2b": BNCI2014_004(),
    "BNCI2015_001": BNCI2015_001(),
    "BNCI2014_002": BNCI2014_002(),
    "Kalunga2016": Kalunga2016(),
    "Nakanishi2015": Nakanishi2015(),
    "BNCI2014_008": BNCI2014_008(),
    "Weibo2014": Weibo2014(),
    "AlexMI": AlexMI(),
    "Zhou2016": Zhou2016(),
    "Cho2017": Cho2017(),
    "BI2015a": BI2015a(),
    "Lee2019_SSVEP": Lee2019_SSVEP(),
    "Lee2019_ERP": Lee2019_ERP(),
    "Lee2019_MI": Lee2019_MI(),
}

PARA_DICT = {
    "2b": LeftRightImagery(),
    "BNCI2015_001": MotorImagery(),
    "BNCI2014_002": MotorImagery(),
    "Kalunga2016": SSVEP(n_classes=4),
    "Nakanishi2015": SSVEP(n_classes=12),
    "BNCI2014_008": P300(),
    "Weibo2014": MotorImagery(),
    "AlexMI": MotorImagery(),
    "Zhou2016": MotorImagery(),
    "Cho2017": MotorImagery(),
    "BI2015a": P300(),
    "Lee2019_SSVEP": SSVEP(n_classes=4),
    "Lee2019_ERP": P300(),
    "Lee2019_MI": MotorImagery(),
}

# -------------------------
# Helpers
# -------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _estimate_map_size(n_records: int, x_shape: Tuple[int, ...], dtype: np.dtype, safety: float = 2.2) -> int:
    """
    粗略估算 map_size（会偏大一点，减少 MapFullError）。
    x_shape: 单条 record 的形状，比如 (C, P, L)
    """
    bytes_per_record = int(np.prod(x_shape) * np.dtype(dtype).itemsize)
    total = int(n_records * bytes_per_record * safety)
    return max(total, 1 * 1024**3)


def _set_paradigm_resample(paradigm, target_sfreq: int) -> None:
    if not hasattr(paradigm, "resample"):
        raise RuntimeError(
            f"Paradigm {type(paradigm).__name__} has no attribute 'resample'. "
            f"Cannot enforce resampling to {target_sfreq} Hz."
        )
    try:
        paradigm.resample = target_sfreq
    except Exception as e:
        raise RuntimeError(
            f"Failed to set paradigm.resample = {target_sfreq}. "
            f"MOABB paradigm does not allow post-hoc resample setting."
        ) from e

    if getattr(paradigm, "resample", None) != target_sfreq:
        raise RuntimeError(
            f"paradigm.resample was not set correctly. "
            f"Expected {target_sfreq}, got {getattr(paradigm, 'resample', None)}."
        )

    print(f"=====================成功重采样到{target_sfreq}Hz!=====================")


def _verify_resample_from_shape(X: np.ndarray, target_sfreq: int, ds_name: str) -> None:
    print(f"X.shape: {X.shape}")
    assert abs(X.shape[-1] - target_sfreq * SEQ_LEN_DICT[ds_name]) <= 2
    print(f"=====================X.shape: {X.shape} 确认成功重采样到{target_sfreq}===================== \n")

def _get_subject_list(ds_name: str) -> List[int]:
    # SUB_NUM_DICT[ds_name] 存的是 subject 数量
    return list(range(1, SUB_NUM_DICT[ds_name] + 1))


def _put_pickle_with_auto_resize(db: lmdb.Environment, txn: lmdb.Transaction, key_b: bytes, obj) -> lmdb.Transaction:
    val_b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        txn.put(key_b, val_b)
        return txn
    except lmdb.MapFullError:
        txn.abort()
        old_size = db.info()["map_size"]
        new_size = int(old_size * 2)
        db.set_mapsize(new_size)
        print(f"[LMDB] MapFullError: map_size {old_size} -> {new_size} (doubled)")
        txn = db.begin(write=True)
        txn.put(key_b, val_b)
        return txn


def _trial_to_patches(x_i: np.ndarray, patch_len: int) -> np.ndarray:
    """
    x_i: (C, T) -> (C, P, patch_len), 丢弃尾部不足一个 patch 的部分
    """
    C, T = x_i.shape
    P = T // patch_len
    if P <= 0:
        raise RuntimeError(f"Trial length T={T} < patch_len={patch_len}. No patches can be formed.")
    x_i = x_i[:, :P * patch_len]                 # (C, P*L)
    x_i = x_i.reshape(C, P, patch_len)           # (C, P, L)
    return np.ascontiguousarray(x_i)


def convert_moabb_to_lmdb_trial_level(
    ds_name: str,
    out_lmdb_dir: str,
    commit_every: int = 2048,
    x_dtype: str = "float32",
    target_sfreq: int = 200,
    patch_len: int = 200,
):
    """
    LMDB 最小单位：一个 trial -> (C, Patch_num, Patch_len)
    兼容旧读取：写 '__keys__'，value = pickle.dumps(np.ndarray)
    """
    assert ds_name in DATA_DICT, f"Unknown dataset: {ds_name}. Available: {list(DATA_DICT.keys())}"
    dataset = DATA_DICT[ds_name]
    paradigm = PARA_DICT[ds_name]

    _ensure_dir(out_lmdb_dir)

    _set_paradigm_resample(paradigm, target_sfreq=target_sfreq)

    dtype = np.dtype(x_dtype)
    subjects = _get_subject_list(ds_name)

    first_sub = subjects[0]
    print(f"[INFO] subjects: {len(subjects)}, first subject: {first_sub}")

    X0, _, _ = paradigm.get_data(dataset=dataset, subjects=[first_sub])
    epochs0, _, _ = paradigm.get_data(
        dataset=dataset,
        subjects=[first_sub],
        return_epochs=True
    )
    print('通道列表：', epochs0.ch_names)
    _verify_resample_from_shape(X0, target_sfreq=target_sfreq, ds_name=ds_name)

    X0 = X0.astype(dtype, copy=False)
    n_trials0, C0, T0 = X0.shape

    # trial-level record shape uses P0 from first subject
    P0 = T0 // patch_len
    if P0 <= 0:
        raise RuntimeError(f"Trial length T={T0} < patch_len={patch_len}. No patches can be formed.")
    record_shape = (C0, P0, patch_len)

    total_records_est = int(len(subjects) * n_trials0)  # 每 trial 一条记录
    map_size = _estimate_map_size(total_records_est, record_shape, dtype=dtype, safety=2.2)

    print(f"[INFO] (estimate) per-sub: trials={n_trials0}, C={C0}, T={T0}, patches/trial={P0}")
    print(f"[INFO] (estimate) total trials ~ {total_records_est}, initial map_size={map_size / (1024**3):.2f} GB")

    db = lmdb.open(out_lmdb_dir, map_size=map_size, subdir=True, meminit=False, lock=True)
    txn = db.begin(write=True)

    keys = []
    written = 0

    def _write_subject_trials(X_sub: np.ndarray, sub_id: int, txn_: lmdb.Transaction, written_: int):
        # (n_trials, C, T)
        n_trials, C, T = X_sub.shape
        print(f"###########################################################################")
        print(f"==============================sub{sub_id}: {n_trials} trials=============================")
        print(f"###########################################################################")

        if C != C0:
            raise RuntimeError(f"Channel mismatch for subject {sub_id}: got C={C}, expected C0={C0}")

        P = T // patch_len
        if P != P0:
            # 如果不同 subject 的 T 有差异，会导致 P 不一致；这会破坏“最小单位固定 shape”的假设
            raise RuntimeError(
                f"Patch_num mismatch for subject {sub_id}: got P={P}, expected P0={P0}. "
                f"(T={T}, patch_len={patch_len})."
            )

        for i in range(n_trials):
            x_i = X_sub[i]  # (C, T)
            x_trial = _trial_to_patches(x_i, patch_len=patch_len)  # (C, P, L)

            if i % 100 == 0:
                print('sampel shape: ', x_trial.shape)

            key = f"{ds_name}_sub{sub_id:03d}_trial{i:06d}"
            txn_ = _put_pickle_with_auto_resize(db, txn_, key.encode(), x_trial)

            keys.append(key)
            written_ += 1

            if written_ % commit_every == 0:
                txn_.commit()
                txn_ = db.begin(write=True)
                print(f"[LMDB] written {written_} trials")

        return txn_, written_

    # write first subject
    txn, written = _write_subject_trials(X0, first_sub, txn, written)
    del X0
    gc.collect()

    for sub in tqdm(subjects[1:]):
        print(f"[INFO] processing subject {sub} ...")
        X_sub, _, _ = paradigm.get_data(dataset=dataset, subjects=[sub])
        _verify_resample_from_shape(X_sub, target_sfreq=target_sfreq, ds_name=ds_name)

        X_sub = X_sub.astype(dtype, copy=False)
        txn, written = _write_subject_trials(X_sub, sub, txn, written)
        print(written)

        del X_sub
        gc.collect()

    txn.put(b"__keys__", pickle.dumps(keys, protocol=pickle.HIGHEST_PROTOCOL))
    txn.commit()
    db.sync()
    db.close()

    print(f"[DONE] LMDB saved to: {out_lmdb_dir}")
    print(f"[DONE] logical dataset shape (trial-level): ({written}, {C0}, {P0}, {patch_len})")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATA_DICT.keys()))
    parser.add_argument("--commit_every", type=int, default=2048)
    parser.add_argument("--x_dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--sfreq", type=int, default=200)
    parser.add_argument("--patch_len", type=int, default=200)
    args = parser.parse_args()

    out_lmdb_dir = os.path.join("/data1/hust_bciml_eegdata/lmdb/", f"{args.dataset}-lmdb")

    convert_moabb_to_lmdb_trial_level(
        ds_name=args.dataset,
        out_lmdb_dir=out_lmdb_dir,
        commit_every=args.commit_every,
        x_dtype=args.x_dtype,
        target_sfreq=args.sfreq,
        patch_len=args.patch_len,
    )


if __name__ == "__main__":
    main()
