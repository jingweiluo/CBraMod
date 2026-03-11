import pickle

import lmdb
from torch.utils.data import Dataset

from utils.util import to_tensor


class PretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dir
    ):
        super(PretrainingDataset, self).__init__()
        self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.dataset_dir = dataset_dir
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))
        # self.keys = self.keys[:100000]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))

        patch = to_tensor(patch)

        # # per-sample, per-channel z-score over (P,S)
        # mean = patch.mean(dim=(1, 2), keepdim=True)                 # (C,1,1)
        # std = patch.std(dim=(1, 2), keepdim=True, unbiased=False)   # (C,1,1)
        # patch = (patch - mean) / (std + 1e-6)

        # print(patch.shape)
        if "Chisco" in self.dataset_dir:
            return patch * 1000000 / 100 # Chisco dataset use Volt unit, others use mu Volt
        else:
            return patch / 100



