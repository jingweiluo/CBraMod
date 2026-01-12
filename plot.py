from utils.util import get_ch_coord
import torch

coords = get_ch_coord('TUEG')
print('original 3d', coords)

c = coords - coords.mean(dim=0, keepdim=True)
# scale by max radius
r = torch.linalg.norm(c, dim=-1)
max_r = torch.clamp(r.max(), min=1e-6)
c = c / max_r  # now roughly within [-1,1] scale

print('norm 3d', c)


# import numpy as np
# import scipy
# from scipy import signal
# import os
# import lmdb
# import pickle
# from scipy.signal import butter, lfilter, resample, filtfilt

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# import time
# import numpy as np
# import mne


# def save_ct_mne_raw_timestamp(
#     x,
#     fs=200,
#     ch_names=None,
#     prefix="eeg_mne",
#     duration=10.0,
#     start=0.0,
#     n_channels=None,
#     figsize=(14, 8),
#     dpi=150,
# ):
#     """
#     用 MNE 把 (C,T) 的数据画成 Raw EEG，并保存到当前目录（时间戳命名）。
#     不对 x 做任何预处理：不归一化、不缩放、不滤波、不重参考。

#     Parameters
#     ----------
#     x : np.ndarray or torch.Tensor, shape (C, T)
#     fs : float
#         sampling frequency (Hz)
#     ch_names : list[str] | None
#         通道名列表，长度为 C；若 None 则自动生成 Ch0..Ch{C-1}
#     prefix : str
#         文件名前缀
#     duration : float
#         Raw 浏览窗口显示的时间长度（秒）
#     start : float
#         从第几秒开始显示（秒）
#     n_channels : int | None
#         显示多少个通道；None 表示全部
#     figsize : tuple
#         图片尺寸
#     dpi : int
#         输出分辨率

#     Returns
#     -------
#     save_path : str
#     """
#     if hasattr(x, "detach"):
#         x = x.detach().cpu().numpy()
#     x = np.asarray(x)

#     assert x.ndim == 2, f"Expected (C,T), got shape {x.shape}"
#     C, T = x.shape

#     if ch_names is None:
#         ch_names = [f"Ch{c}" for c in range(C)]
#     assert len(ch_names) == C, f"ch_names length {len(ch_names)} != C {C}"

#     # MNE RawArray expects shape (n_channels, n_times)
#     info = mne.create_info(ch_names=ch_names, sfreq=float(fs), ch_types=["eeg"] * C)
#     raw = mne.io.RawArray(x, info, verbose="ERROR")

#     # 生成时间戳文件名（保存到当前目录）
#     ts = time.strftime("%Y%m%d_%H%M%S")
#     save_path = f"{prefix}_{ts}.png"

#     # 画图（show=False），然后保存
#     fig = raw.plot(
#         start=float(start),
#         duration=float(duration),
#         n_channels=int(n_channels) if n_channels is not None else C,
#         show=False,
#         scalings=dict(eeg=1.0),  # 关键：不做额外缩放（显示时的“每单位值”就是你的原始数值）
#         remove_dc=False,         # 不去直流分量（避免“预处理”）
#         block=False,
#     )

#     # MNE 的浏览图对象支持 savefig
#     fig.set_size_inches(figsize[0], figsize[1], forward=True)
#     fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
#     fig.close()

#     return save_path

# def butter_bandpass(low_cut, high_cut, fs, order=5):
#     nyq = 0.5 * fs
#     low = low_cut / nyq
#     high = high_cut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# root_dir = '/data1/hust_bciml_eegdata/BCICIV-2a-mat'
# files = [file for file in os.listdir(root_dir)]
# files = sorted(files)

# print(files)

# files_dict = {
#     'train': ['A01E.mat', 'A01T.mat', 'A02E.mat', 'A02T.mat', 'A03E.mat', 'A03T.mat',
#               'A04E.mat', 'A04T.mat',
#               'A05E.mat', 'A05T.mat'],
#     'val': [
#         'A06E.mat', 'A06T.mat',
#         'A07E.mat', 'A07T.mat'
#     ],
#     'test': ['A08E.mat', 'A08T.mat', 'A09E.mat', 'A09T.mat'],
# }


# dataset = {
#     'train': list(),
#     'val': list(),
#     'test': list(),
# }


# for files_key in files_dict.keys():
#     for file in files_dict[files_key]:
#         # print(file)
#         data = scipy.io.loadmat(os.path.join(root_dir, file))
#         num = len(data['data'][0])
#         # print(num)
#         # print(data['data'][0, 8][0, 0][0].shape)
#         # print(data['data'][0, 8][0, 0][1].shape)
#         # print(data['data'][0, 8][0, 0][2].shape)
#         for j in range(3, num):
#             raw_data = data['data'][0, j][0, 0][0][:, :22]
#             events = data['data'][0, j][0, 0][1][:, 0]
#             labels = data['data'][0, j][0, 0][2][:, 0]
#             length = raw_data.shape[0]
#             events = events.tolist()
#             events.append(length)
#             # print(events)
#             annos = []
#             for i in range(len(events) - 1):
#                 annos.append((events[i], events[i + 1]))
#             for i, (anno, label) in enumerate(zip(annos, labels)):
#                 sample = raw_data[anno[0]:anno[1]].transpose(1, 0)
#                 sample  = sample - np.mean(sample, axis=0, keepdims=True)

#                 # print(samples.shape)
#                 b, a = butter_bandpass(0.3, 50, 250)
#                 sample = lfilter(b, a, sample, -1)
#                 # print(sample.shape)
#                 sample = sample[:, 2 * 250:6 * 250]
#                 sample = resample(sample, 800, axis=-1)

#                 if j == 4 and i == 0:
#                     print(sample)

#                 # print(sample.shape)
#                 # print(i, sample.shape, label)
#                 sample = sample.reshape(22, 4, 200)
#                 sample_key = f'{file[:-4]}-{j}-{i}'
#                 # print(sample_key, label-1)
#                 data_dict = {
#                     'sample': sample, 'label': label - 1
#                 }
#                 # print(label-1)




