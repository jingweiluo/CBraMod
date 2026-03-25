import argparse
import random

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset
from finetune_trainer import Trainer
# from models import model_for_faced, model_for_faced26, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
#     model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
#     model_for_bciciv2a
from models import downstream_classifier

from utils.constants import LMDB_DIR_DICT, CLS_NUM_DICT, ROOT_DIR, ROOT_DIR2, CHAN_NAME_DICT, SEQ_LEN_DICT
import os


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='BCIC-IV-2a',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a, FACED26]')
    # parser.add_argument('--datasets_dir', type=str,
    #                     default='/data1/labram_data/BCICIV-2a-mat', # /data1/labram_data/faced/faced-cbra-lmdb/
    #                     help='datasets_dir')
    # parser.add_argument('--num_of_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='finetune_model', help='model_dir')
    parser.add_argument('--log_file_name', type=str, default='test_results.txt', help='log_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument(
        '--frozen',
        action='store_true',
        help='frozen'
    )
    parser.add_argument(
        '--use_pretrained_weights',
        action='store_true',
        help='use pretrained encoder weights'
    )
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()

    # 根据数据集名称自动读取相关信息
    dataset_name = params.downstream_dataset
    root = ROOT_DIR2 if dataset_name in ['SHU-MI', 'MentalArithmetic', 'CHB-MIT', 'Mumtaz2016'] else ROOT_DIR
    params.datasets_dir = os.path.join(root, 'lmdb', LMDB_DIR_DICT[dataset_name])
    params.num_of_classes = CLS_NUM_DICT[dataset_name]
    params.seq_len = SEQ_LEN_DICT[dataset_name]

    # 计算出3d坐标
    from utils.util import get_ch_coord
    params.ch_names = CHAN_NAME_DICT[dataset_name]
    params.ch_coords = get_ch_coord(dataset_name)  # (C,3)


    print(params)
    print('是否使用预训练模型', params.use_pretrained_weights)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'FACED':
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SEED-V':
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'PhysioNet-MI':
        load_dataset = physio_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC':
        load_dataset = isruc_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'Mumtaz2016':
        load_dataset = mumtaz_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'SEED-VIG':
        load_dataset = seedvig_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_regression()
    elif params.downstream_dataset == 'MentalArithmetic':
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC-IV-2a':
        load_dataset = bciciv2a_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = downstream_classifier.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()