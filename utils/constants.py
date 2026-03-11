ROOT_DIR='/data1/hust_bciml_eegdata/'

MOABB_DATASET_LIST = [
    # e.g. "BNCI2014_001", "PhysionetMI", "Cho2017", "Lee2019_SSVEP"
    '2b', 'BNCI2015_001', 'BNCI2014_008',
    'Kalunga2016', 'Nakanishi2015', 'Weibo2014',
    'Zhou2016', 'AlexMI', 'Cho2017',
    'BI2015a', 'Lee2019_SSVEP', 'Lee2019_ERP', 'Lee2019_MI',
]

CLS_NUM_DICT = {
    'BCIC-IV-2a': 4,
    '2b': 2,
    'BNCI2015_001': 2,
    'BNCI2014_002': 2,
    'Kalunga2016': 4,
    'Nakanishi2015': 12,
    'BNCI2014_008': 2,
    'Weibo2014': 2,
    'Zhou2016': 2,
    'AlexMI': 3,
    'TUAB': 2,
    'PhysioNet-MI': 4, # left,right,both fists,both feet
    'FACED': 9,
    'FACED26': 9,
    'BCIC2020-3': 5,
    'Cho2017': 2,
    'BI2015a': 2,
    'Lee2019_SSVEP': 4,
    'Lee2019_ERP': 2,
    'Lee2019_MI': 2,
    'Chisco': 39,
    'Chisco_read': 39,
}

SEQ_LEN_DICT = {
    'BCIC-IV-2a': 4,
    '2b': 4.5,
    'BNCI2015_001': 5,
    'BNCI2014_002': 5,
    'Kalunga2016': 2,
    'Nakanishi2015': 4.15,
    'BNCI2014_008': 1,
    'Weibo2014': 2,
    'Zhou2016': 5,
    'AlexMI': 3,
    'TUEG': 30,
    'TUAB': 2,
    'PhysioNet-MI': 4, # left,right,both fists,both feet
    'FACED': 10,
    'FACED26': 10,
    'BCIC2020-3': 3,
    'Cho2017': 3,
    'BI2015a': 1,
    'Lee2019_SSVEP': 4,
    'Lee2019_ERP': 1,
    'Lee2019_MI': 4,
    'Chisco': 3,
    'Chisco_read': 5,
}

SUB_NUM_DICT = {
    'BCIC-IV-2a': 9,
    '2b': 9,
    'BNCI2015_001': 9,
    'BNCI2014_002': 14,
    'Kalunga2016': 12,
    'Nakanishi2015': 9, # 一共180trials
    'BNCI2014_008': 8,
    'Weibo2014': 10,
    'Zhou2016': 4,
    'AlexMI': 8,
    'PhysioNetMI': 109,
    'FACED': 123,
    'FACED26': 123,
    'Cho2017': 52,
    'BI2015a': 43,
    'Lee2019_SSVEP': 54,
    'Lee2019_ERP': 54,
    'Lee2019_MI': 54,
    'Chisco': 5,
    'Chisco_read': 5,
}

CHAN_NAME_DICT = {
    '2b': ["C3", "Cz", "C4"], # 3
    'BCIC-IV-2a': ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"], # 22
    'TUEG': ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"], # 19
    'BNCI2015_001': ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'], # 13
    'Zhou2016': ["FP1","FP2","FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4","O1","Oz","O2"], # 14
    'Nakanishi2015': ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"], # 8
    'AlexMI': ["Fpz", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8"], # 16
    'TUAB': ["FP1","F7","T3","T5","O1","FP2","F8","T4","T6","O2","F3","C3","P3","F4","C4","P4"], # 16
    'PhysioNet-MI': ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6","Fp1","Fpz","Fp2","AF7","AF3","AFz","AF4","AF8","F7","F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8","T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1","Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8","O1","Oz","O2","Iz"], # 64
    'FACED': [
        'Fp1','Fp2','Fz','F3','F4','F7','F8',
        'FC1','FC2','FC5','FC6',
        'Cz','C3','C4', 'T7', 'T8',
        'CP1','CP2','CP5','CP6',
        'Pz','P3','P4', 'P7', 'P8',
        'PO3','PO4','Oz','O1','O2', 'A1', 'A2',
    ], #32 common chans
    'FACED26': [
        'Fp1','Fp2','Fz','F3','F4','F7','F8',
        'FC1','FC2','FC5','FC6',
        'Cz','C3','C4',
        'CP1','CP2','CP5','CP6',
        'Pz','P3','P4',
        'PO3','PO4','Oz','O1','O2',
    ], #26 common chans
    'BCIC2020-3': [
        "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6",
        "T7","C3","Cz","C4","T8","TP9","CP5","CP1","CP2","CP6",
        "TP10","P7","P3","Pz","P4","P8","PO9","O1","Oz","O2",
        "PO10","AF7","AF3","AF4","AF8","F5","F1","F2","F6",
        "FT9","FT7","FC3","FC4","FT8","FT10","C5","C1","C2","C6",
        "TP7","CP3","CPz","CP4","TP8","P5","P1","P2","P6",
        "PO7","PO3","POz","PO4","PO8"
    ],
    'BNCI2014_008': ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"],
    'Cho2017': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
    'BI2015a': ['Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9', 'PO10'],
    'Lee2019_SSVEP': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
    'Lee2019_ERP': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'],
    'Lee2019_MI': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4'], # 论文
    'Kalunga2016': ['Oz', 'O1', 'O2', 'PO3', 'POz', 'PO7', 'PO8', 'PO4'], # moabb直接读取
    'Chisco': ['T9', 'FT9', 'FTT9h', 'T7', 'TP7', 'TTP7h', 'C5', 'FTT7h', 'Ft7', 'FFT7h', 'FC5', 'FCC5h', 'CCP5h', 'TPP5h', 'P7', 'P9', 'P11', 'PO11', 'PO9', 'PPO7', 'P5', 'CPP5h', 'CP3', 'C3', 'FC3', 'FFC5h', 'F5', 'F7', 'AF7', 'AFF5h', 'F3', 'FFC3h', 'FCC3h', 'C1', 'CCP3h', 'CPP3h', 'P3', 'PO3', 'POO7', 'POO9h', 'POO11h', 'I1', 'OI1', 'POO3', 'PO1', 'PPO1', 'CPP1h', 'CP1', 'CCP1h', 'FCC1h', 'FC1', 'FFC1h', 'F1', 'AF3', 'FP1', 'FPz', 'AFz', 'Fz', 'FCz', 'FCCz', 'Cz', 'CPPz', 'PPOz', 'POz', 'POOz', 'Oz', 'Iz', 'I2', 'OI2', 'POO4', 'PO2', 'PPO2', 'CPP2h', 'CP2', 'CCP2h', 'FCC2h', 'FC2', 'FFC2h', 'F2', 'AF4', 'Fp2', 'AF8', 'AFF6h', 'F4', 'FFC4h', 'FCC4h', 'C2', 'CCP4h', 'CPP4h', 'P4', 'PO4', 'POO8', 'POO10h', 'POO12h', 'PO12', 'PO10', 'PPO8', 'P6', 'CPP6h', 'CP4', 'C4', 'FC4', 'FFC6h', 'F6', 'F8', 'FFT8h', 'FC6', 'FCC6h', 'CCP6h', 'TPP8h', 'P8', 'P10', 'P12', 'TTP8h', 'C6', 'FTT8h', 'FT8', 'T10', 'FT10', 'FTT10h', 'T8', 'TP8'],
    'Chisco_read': ['T9', 'FT9', 'FTT9h', 'T7', 'TP7', 'TTP7h', 'C5', 'FTT7h', 'Ft7', 'FFT7h', 'FC5', 'FCC5h', 'CCP5h', 'TPP5h', 'P7', 'P9', 'P11', 'PO11', 'PO9', 'PPO7', 'P5', 'CPP5h', 'CP3', 'C3', 'FC3', 'FFC5h', 'F5', 'F7', 'AF7', 'AFF5h', 'F3', 'FFC3h', 'FCC3h', 'C1', 'CCP3h', 'CPP3h', 'P3', 'PO3', 'POO7', 'POO9h', 'POO11h', 'I1', 'OI1', 'POO3', 'PO1', 'PPO1', 'CPP1h', 'CP1', 'CCP1h', 'FCC1h', 'FC1', 'FFC1h', 'F1', 'AF3', 'FP1', 'FPz', 'AFz', 'Fz', 'FCz', 'FCCz', 'Cz', 'CPPz', 'PPOz', 'POz', 'POOz', 'Oz', 'Iz', 'I2', 'OI2', 'POO4', 'PO2', 'PPO2', 'CPP2h', 'CP2', 'CCP2h', 'FCC2h', 'FC2', 'FFC2h', 'F2', 'AF4', 'Fp2', 'AF8', 'AFF6h', 'F4', 'FFC4h', 'FCC4h', 'C2', 'CCP4h', 'CPP4h', 'P4', 'PO4', 'POO8', 'POO10h', 'POO12h', 'PO12', 'PO10', 'PPO8', 'P6', 'CPP6h', 'CP4', 'C4', 'FC4', 'FFC6h', 'F6', 'F8', 'FFT8h', 'FC6', 'FCC6h', 'CCP6h', 'TPP8h', 'P8', 'P10', 'P12', 'TTP8h', 'C6', 'FTT8h', 'FT8', 'T10', 'FT10', 'FTT10h', 'T8', 'TP8'],
}

SFREQ_DICT = {
    'BCIC-IV-2a': 250,
    '2b': 250,
    'BNCI2015_001': 512,
    'BNCI2014_002': 512,
    'Kalunga2016': 256,
    'Nakanishi2015': 256,
    'BNCI2014_008': 256,
    'Weibo2014': 200,
    'TUEG': 200,
    'Zhou2016': 250,
    'AlexMI': 512,
    'TUAB': 200,
    'PhysioNet-MI': 200,
    'FACED': 200, # originally 250
    'FACED26': 200,
    'BCIC2020-3': 200,
    'Cho2017': 512,
    'BI2015a': 512,
    'Lee2019_SSVEP': 1000,
    'Lee2019_ERP': 1000,
    'Lee2019_MI': 1000,
    'Chisco': 500,
    'Chisco_read': 500,
}

DATA_DIR_DICT = {
    'BCIC-IV-2a': 'BCICIV-2a-mat',
    'TUEG': 'TUEG',
    'TUAB': 'TUAB',
    'TUEV': 'TUEV',
    'PhysioNet-MI': 'PhysioNetMI',
    'FACED': 'FACED/Processed_data',
    'FACED26': 'FACED/Processed_data',
    'BCIC2020-3': 'BCIC2020-3',
    'Chisco': 'Chisco',
    'Chisco_read': 'Chisco_read',
}

LMDB_DIR_DICT = {
    'BCIC-IV-2a': 'BCICIV-2a-lmdb',
    '2b': 'BCICIV-2b-lmdb',
    'TUEG': 'TUEG-lmdb',
    'BNCI2015_001': 'BNCI2015_001-lmdb',
    'Zhou2016': 'Zhou2016-lmdb',
    'Nakanishi2015': 'Nakanishi2015-lmdb',
    'AlexMI': 'AlexMI-lmdb',
    'TUAB': 'TUAB-lmdb',
    'TUEV': 'TUEV-lmdb',
    'PhysioNet-MI': 'PhysioNetMI-lmdb',
    'FACED': 'FACED-lmdb',
    'FACED26': 'FACED26-lmdb',
    'BCIC2020-3': 'BCIC2020-3-lmdb',
    'Lee2019_SSVEP': 'Lee2019_SSVEP-lmdb',
    'Lee2019_ERP': 'Lee2019_ERP-lmdb',
    'Lee2019_MI': 'Lee2019_MI-lmdb',
    'Cho2017': 'Cho2017-lmdb',
    'BI2015a': 'BI2015a-lmdb',
    'BNCI2014_008': 'BNCI2014_008-lmdb',
    'Kalunga2016': 'Kalunga2016-lmdb',
    'Chisco': 'Chisco-lmdb',
    'Chisco_read': 'Chisco_read-lmdb',
}