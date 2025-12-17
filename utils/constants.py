ROOT_DIR='/data1/hust_bciml_eegdata/'

CLS_NUM_DICT = {
    '2a': 4,
    '2b': 2,
    'BNCI2015_001': 2,
    'BNCI2014_002': 2,
    'Kalunga2016': 4,
    'Nakanishi2015': 12,
    'BNCI2014008': 2,
    'Weibo2014': 2,
    'Zhou2016': 2,
    'AlexMI': 3,
    'TUAB': 2,
    'PhysioNet-MI': 4, # left,right,both fists,both feet
    'FACED': 9,
    'BCIC2020-3': 5,
}

# SUB_NUM_DICT = {
#     '2a': 9,
#     '2b': 9,
#     'BNCI2015_001': 9,
#     'BNCI2014_002': 14,
#     'Kalunga2016': 12,
#     'Nakanishi2015': 9, # 一共180trials
#     'BNCI2014008': 8,
#     'Weibo2014': 10,
#     'Zhou2016': 4,
#     'AlexMI': 8,
#     'PhysioNetMI': 109,
#     'FACED': 123,
#     'BCIC2020-3': ,
# }

CHAN_NAME_DICT = {
    '2b': ["C3", "Cz", "C4"], # 3
    '2a': ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"], # 22
    'TUEG': ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2","F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"], # 19
    'BNCI2015_001': ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4"], # 13
    'BNCI2014_002': [],
    'Zhou2016': ["FP1","FP2","FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4","O1","Oz","O2"], # 14
    'Nakanishi2015': ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"], # 8
    'AlexMI': ["Fpz", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8"], # 16
    'TUAB': ["FP1","F7","T3","T5","O1","FP2","F8","T4","T6","O2","F3","C3","P3","F4","C4","P4"], # 16
    'PhysioNet-MI': ["FC5","FC3","FC1","FCz","FC2","FC4","FC6","C5","C3","C1","Cz","C2","C4","C6","CP5","CP3","CP1","CPz","CP2","CP4","CP6","Fp1","Fpz","Fp2","AF7","AF3","AFz","AF4","AF8","F7","F5","F3","F1","Fz","F2","F4","F6","F8","FT7","FT8","T7","T8","T9","T10","TP7","TP8","P7","P5","P3","P1","Pz","P2","P4","P6","P8","PO7","PO3","POz","PO4","PO8","O1","Oz","O2","Iz"], # 64
    'FACED': [
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
}

SFREQ_DICT = {
    '2a': 250,
    '2b': 250,
    'BNCI2015_001': 512,
    'BNCI2014_002': 512,
    'Kalunga2016': 256,
    'Nakanishi2015': 256,
    'BNCI2014008': 256,
    'Weibo2014': 200,
    'TUEG': 200,
    'Zhou2016': 250,
    'AlexMI': 512,
    'TUAB': 200,
    'PhysioNet-MI': 200,
    'FACED': 200,
    'BCIC2020-3': 200,
}

DATA_DIR_DICT = {
    '2a': 'BCICIV-2a-mat',
    'TUEG': 'TUEG',
    'TUAB': 'TUAB',
    'TUEV': 'TUEV',
    'PhysioNet-MI': 'PhysioNetMI',
    'FACED': 'FACED/Processed_data',
    'BCIC2020-3': 'BCIC2020-3',
}

LMDB_DIR_DICT = {
    '2a': 'BCICIV-2a-lmdb',
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
    'BCIC2020-3': 'BCIC2020-3-lmdb',
}