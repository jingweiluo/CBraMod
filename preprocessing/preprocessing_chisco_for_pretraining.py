import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb

ch_list = ['T9', 'FT9', 'FTT9h', 'T7', 'TP7', 'TTP7h', 'C5', 'FTT7h', 'Ft7', 'FFT7h', 'FC5', 'FCC5h', 'CCP5h', 'TPP5h', 'P7', 'P9', 'P11', 'PO11', 'PO9', 'PPO7', 'P5', 'CPP5h', 'CP3', 'C3', 'FC3', 'FFC5h', 'F5', 'F7', 'AF7', 'AFF5h', 'F3', 'FFC3h', 'FCC3h', 'C1', 'CCP3h', 'CPP3h', 'P3', 'PO3', 'POO7', 'POO9h', 'POO11h', 'I1', 'OI1', 'POO3', 'PO1', 'PPO1', 'CPP1h', 'CP1', 'CCP1h', 'FCC1h', 'FC1', 'FFC1h', 'F1', 'AF3', 'FP1', 'FPz', 'AFz', 'Fz', 'FCz', 'FCCz', 'Cz', 'CPPz', 'PPOz', 'POz', 'POOz', 'Oz', 'Iz', 'I2', 'OI2', 'POO4', 'PO2', 'PPO2', 'CPP2h', 'CP2', 'CCP2h', 'FCC2h', 'FC2', 'FFC2h', 'F2', 'AF4', 'Fp2', 'AF8', 'AFF6h', 'F4', 'FFC4h', 'FCC4h', 'C2', 'CCP4h', 'CPP4h', 'P4', 'PO4', 'POO8', 'POO10h', 'POO12h', 'PO12', 'PO10', 'PPO8', 'P6', 'CPP6h', 'CP4', 'C4', 'FC4', 'FFC6h', 'F6', 'F8', 'FFT8h', 'FC6', 'FCC6h', 'CCP6h', 'TPP8h', 'P8', 'P10', 'P12', 'TTP8h', 'C6', 'FTT8h', 'FT8', 'T10', 'FT10', 'FTT10h', 'T8', 'TP8']
error_count = 0


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

#遍历文件夹
def iter_files(rootDir):
    #遍历根目录
    file_path_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            if file.lower().endswith(".fif") and "task-read" in file.lower():
                file_name = os.path.join(root, file)
                file_path_list.append(file_name)
    return file_path_list

def preprocessing_recording(file_path, file_key_list: list, db: lmdb.open):
    global error_count
    epochs = mne.read_epochs(file_path, preload=True, verbose="ERROR")

    # --------- 简单检查1：通道是否齐全（不足122则跳过该文件）---------
    exist = set(epochs.ch_names)
    picks = [ch for ch in ch_list if ch in exist]
    if len(picks) < 122:
        error_count = error_count + 1
        print(f'检查到不符合标准的文件: 通道缺失,只有{len(picks)}通道，当前error_count:{error_count}')
        return  # 跳过

    epochs.pick_channels(picks, ordered=True)
    epochs.resample(200)
    data = epochs.get_data()  # (n_trials, 122, n_times)

    # --------- 简单检查2：时间长度是否足够（不足600则跳过该文件）---------
    if data.shape[-1] < 1000:
        error_count += 1
        print(f'检查到不符合标准的文件: 长度缺失,只有{data.shape[-1]}points,当前error_count:{error_count}')
        return  # 跳过

    raw = data[:, :, :1000]  # (n_trials, 122, 600)
    num_trials, chs, points = raw.shape
    raw = raw.reshape(num_trials, chs, 5, 200)
    file_name = file_path.split('/')[-1][:-4]

    txn = db.begin(write=True)
    for i, sample in enumerate(raw):
        sample_key = f'{file_name}_{i}'
        file_key_list.append(sample_key)
        txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
    txn.commit()

if __name__ == '__main__':
    setup_seed(1)
    file_path_list = iter_files('/data1/hust_bciml_eegdata/Chisco/derivatives/preprocessed_fif/')
    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    db = lmdb.open('/data1/hust_bciml_eegdata/lmdb/Chisco_read-lmdb', map_size=50 * 1024**3)
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, db)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()
    print('不符合标准的文件数量', error_count)
