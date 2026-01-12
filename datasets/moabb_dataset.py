import numpy as np
from torch.utils.data import Dataset
from utils.util import to_tensor

from moabb.paradigms import LeftRightImagery, MotorImagery, P300, SSVEP
from moabb.datasets import (
    BNCI2014_001,
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
from utils.constants import SFREQ_DICT
from scipy.signal import resample_poly

DATA_DICT = {
    '2b': BNCI2014_004(),
    'BNCI2015_001': BNCI2015_001(),
    'BNCI2014_002': BNCI2014_002(),
    'Kalunga2016': Kalunga2016(),
    'Nakanishi2015': Nakanishi2015(),
    'BNCI2014_008': BNCI2014_008(),
    'Weibo2014': Weibo2014(),
    'AlexMI': AlexMI(),
    'Zhou2016': Zhou2016(),
    'Cho2017': Cho2017(),
    'BI2015a': BI2015a(),
    'Lee2019_SSVEP': Lee2019_SSVEP(),
    'Lee2019_ERP': Lee2019_ERP(),
    'Lee2019_MI': Lee2019_MI(),
}

PARA_DICT = {
    '2b': LeftRightImagery(),
    'BNCI2015_001': MotorImagery(),
    'BNCI2014_002': MotorImagery(),
    'Kalunga2016': SSVEP(n_classes=4),
    'Nakanishi2015': SSVEP(n_classes=12),
    'BNCI2014_008': P300(),
    'Weibo2014': MotorImagery(),
    'AlexMI': MotorImagery(),
    'Zhou2016': MotorImagery(),
    'Cho2017': MotorImagery(),
    'BI2015a': P300(),
    'Lee2019_SSVEP': SSVEP(n_classes=4),
    'Lee2019_ERP': P300(),
    'Lee2019_MI': MotorImagery(),
}


class MoabbPretrainingDataset(Dataset):
    def __init__(self, dataset_name: str, overlap=0.5):
        super().__init__()
        self.dataset_name = str(dataset_name)

        ds = DATA_DICT[self.dataset_name]
        paradigm = PARA_DICT[self.dataset_name]

        X, y, meta = paradigm.get_data(dataset=ds)  # 关键修正：传 dataset 实例
        X = np.asarray(X)

        # X: (N, C, T)
        if X.ndim != 3:
            n, c = int(X.shape[0]), int(X.shape[1])
            X = X.reshape(n, c, -1)

        X = X.astype(np.float32, copy=False)

        # ---------- resample to 200 Hz ----------
        orig_sfreq = float(SFREQ_DICT[dataset_name])
        if int(round(orig_sfreq)) == 200:
            X200 = X
            sfreq = 200
        else:
            up = 200
            down = int(round(orig_sfreq))
            X200 = resample_poly(X, up=up, down=down, axis=-1).astype(np.float32, copy=False)
            sfreq = 200

        # ---------- sliding window: W=200 (1s) ----------
        W = 200
        if isinstance(overlap, float):
            step = max(1, int(round(W * (1.0 - overlap))))
        else:
            step = max(1, int(W - int(overlap)))

        T = int(X200.shape[-1])
        starts = list(range(0, T - W + 1, step))
        P = len(starts)

        # (N, C, P, W)
        Xp = np.stack([X200[..., s:s + W] for s in starts], axis=2).astype(np.float32, copy=False)

        self.X = Xp
        self.y = y
        self.meta = meta

        self.keys = list(range(int(self.X.shape[0])))

        self.n_channels = int(self.X.shape[1])
        self.seq_len = int(self.X.shape[2])   # P
        self.patch_width = int(self.X.shape[3])  # W=200

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[int(idx)]
        patch = self.X[k]  # (C, P, W)
        patch = to_tensor(patch)
        return patch / 100
