import os
import re
import lmdb
import pickle
import numpy as np
from scipy import signal

# -----------------------------
# Labels (unchanged)
# -----------------------------
labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])

root_dir = '/data1/hust_bciml_eegdata/FACED/Processed_data/'
files = sorted([file for file in os.listdir(root_dir)])

files_dict = {
    'train': files[:80],
    'val':   files[80:100],
    'test':  files[100:],
}

dataset = {'train': [], 'val': [], 'test': []}

# -----------------------------
# Channel lists from your screenshot
# -----------------------------
COHORT1_CHANS_32 = [
    'Fp1','Fp2','Fz','F3','F4','F7','F8','FC1',
    'FC2','FC5','FC6','Cz','C3','C4','T3','T4',
    'A1','A2','CP1','CP2','CP5','CP6','Pz','P3',
    'P4','T5','T6','PO3','PO4','Oz','O1','O2',
]

COHORT2_CHANS_32 = [
    'Fp1','Fp2','Fz','F3','F4','F7','F8','FC1',
    'FC2','FC5','FC6','Cz','C3','C4','T7','T8',
    'CP1','CP2','CP5','CP6','Pz','P3','P4','P7',
    'P8','PO3','PO4','Oz','O1','O2','HEOR','HEOL',
]

# 26 common chans (your requirement)
COMMON_26 = [
    'Fp1','Fp2','Fz','F3','F4','F7','F8',
    'FC1','FC2','FC5','FC6',
    'Cz','C3','C4',
    'CP1','CP2','CP5','CP6',
    'Pz','P3','P4',
    'PO3','PO4','Oz','O1','O2',
]

def parse_sub_id(filename: str):
    """Try parse subXXX from filename; return int or None."""
    m = re.search(r"sub(\d+)", filename)
    return int(m.group(1)) if m else None

def get_sel_idx_for_file(filename: str):
    """
    Decide cohort by sub id:
      - sub000 ~ sub060 => cohort1
      - sub061 ~ sub122 => cohort2
    Return indices of COMMON_26 in the corresponding 32-chan order.
    """
    sid = parse_sub_id(filename)
    if sid is None:
        # fallback: assume cohort1
        ch_order = COHORT1_CHANS_32
    else:
        ch_order = COHORT1_CHANS_32 if sid <= 60 else COHORT2_CHANS_32

    missing = [ch for ch in COMMON_26 if ch not in ch_order]
    if missing:
        raise ValueError(f"Missing common chans {missing} in channel order for file={filename}")

    return [ch_order.index(ch) for ch in COMMON_26]

# -----------------------------
# LMDB
# -----------------------------
db = lmdb.open('/data1/hust_bciml_eegdata/lmdb/FACED26-lmdb/', map_size=6612500172)

for split in files_dict.keys():
    for file in files_dict[split]:
        with open(os.path.join(root_dir, file), 'rb') as f:
            array = pickle.load(f)

        # array expected shape: (28, 32, T) where T resampled to 6000
        eeg = signal.resample(array, 6000, axis=2)          # (28,32,6000)
        eeg_ = eeg.reshape(28, 32, 30, 200)                 # (28,32,30,200)

        # select 26 common channels according to cohort
        sel_idx = get_sel_idx_for_file(file)                # list length 26
        eeg_ = eeg_[:, sel_idx, :, :]                       # (28,26,30,200)

        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                # samples: (26,30,200) -> pick 10 patches each segment => (26,10,200)
                sample = samples[:, 10*j:10*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                data_dict = {'sample': sample, 'label': label}

                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()

                dataset[split].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
