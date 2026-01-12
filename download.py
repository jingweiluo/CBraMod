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
    'Nakanishi2015': SSVEP(n_classes=12), # error
    'BNCI2014_008': P300(),
    'Weibo2014': MotorImagery(), # error
    'AlexMI': MotorImagery(),
    'Zhou2016': MotorImagery(),
    'Cho2017': MotorImagery(),
    'BI2015a': P300(),
    'Lee2019_SSVEP': SSVEP(n_classes=4),
    'Lee2019_ERP': P300(),
    'Lee2019_MI': MotorImagery(),
}


if __name__ == '__main__':
        ds = 'Kalunga2016'
        dataset = DATA_DICT[ds]
        paradigm = PARA_DICT[ds]

        X, y, meta = paradigm.get_data(dataset=dataset)  # 关键修正：传 dataset 实例
        print(X.shape, y.shape)
        print(meta)