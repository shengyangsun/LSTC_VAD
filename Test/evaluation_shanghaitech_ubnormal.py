import os
import sys
sys.path.append('..')
import argparse
import random
import math
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from utils.utils import set_seeds
from utils.utils import get_video_names
from utils.eval_utils import eval
from tqdm import tqdm
from models.Regressor import Regressor
from models.Classifier import Classifier
from utils.load_dataset import SH_Train_Origin_Dataset,shanghaitech_test, UBnormal_test
from models.Encoder import Encoder
from collections import OrderedDict


def evaluation(args):

    '''
        load dataset
    '''
    if args.dataset == "SHT":
        test_feats, test_labels, test_annos, names = shanghaitech_test(args.testing_txt, args.test_mask_dir, args.dataset_path, return_names=True)
    elif args.dataset == "UBnormal":
        test_feats, test_labels, test_annos, names = UBnormal_test(args.testing_txt, args.test_mask_dir, args.dataset_path, return_names=True)
    print ("Dataset load complete.")
    print ("Dataset:", args.dataset)
    state_dict = torch.load(args.temporal_model_path)
    if args.temporal_data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    temporal_model = Encoder(n_layers=args.temporal_n_layers, n_head=args.temporal_n_head, d_k=args.temporal_d_k,
                             d_v=args.temporal_d_v, d_model=args.d_model, d_inner=args.temporal_n_hidden,
                             MHA_layerNorm=args.temporal_MHA_layerNorm, FFN_layerNorm=args.temporal_FFN_layerNorm,
                             relative_pe=args.temporal_relative_position_encoding, window_size=args.window_size,
                             window_depth=args.part_len)

    temporal_model.load_state_dict(new_state_dict, False)
    temporal_model = temporal_model.cuda().eval()

    state_dict = torch.load(args.classifier_model_path)
    if args.temporal_data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    classifier_model = Classifier(args.d_model)
    classifier_model.load_state_dict(new_state_dict, False)
    classifier_model = classifier_model.cuda().eval()

    print ("Model load complete.")


    test_scores_list = []
    test_labels_list = []
    with torch.no_grad():
        for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
            feats_all = torch.from_numpy(np.array(test_feat)[:, :args.n_patch, :]).cuda().float()
            n_clips = feats_all.shape[0]
            total_len = 0
            n_clip_part = n_clips // args.part_len
            if n_clip_part * args.part_len < n_clips:
                n_clip_part += 1
            for clip_part_i in range(n_clip_part):
                beg = clip_part_i * args.part_len
                if clip_part_i == n_clip_part - 1:
                    end = n_clips
                else:
                    end = (clip_part_i + 1) * args.part_len
                if end - beg < args.part_len:
                    feats = feats_all[end - args.part_len:end, :, :].view([1, -1, args.d_model])
                else:
                    feats = feats_all[beg:end, :, :].view([1, -1, args.d_model])
                feats = temporal_model(feats)
                feats = feats[:, 0, :]
                logits = classifier_model(feats).view([-1, 2])
                logits = logits[:, 1]
                score = logits.cpu().numpy()
                test_scores_list.extend([score] * (end - beg) * args.segment_len)
                test_labels_list.extend(test_anno[total_len:total_len + (end - beg) * args.segment_len].tolist())
                total_len += (end - beg) * args.segment_len
        auc_test = eval(test_scores_list, test_labels_list, None)
        print("auc = ", auc_test)
        return


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--dataset', type=str, default='SHT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--part_len', type=int, default=3)

    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=2048)

    parser.add_argument('--temporal_n_head', type=int, default=8)
    parser.add_argument('--temporal_n_hidden', type=int, default=4096)
    parser.add_argument('--temporal_d_k', type=int, default=256)
    parser.add_argument('--temporal_d_v', type=int, default=256)
    parser.add_argument('--temporal_n_layers', type=int, default=3)
    parser.add_argument('--temporal_MHA_layerNorm', action="store_true", help="Run to activate MHA_layerNorm of temporal")
    parser.add_argument('--temporal_FFN_layerNorm', action="store_true", help="Run to activate FFN_layerNorm of temporal")
    parser.add_argument('--temporal_relative_position_encoding', action="store_true", help="Run to activate rpe")
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--temporal_data_parallel', action="store_true", help="training with data_parallel")

    parser.add_argument('--generate_clip_labels', action="store_true", help="Run to generate clip labels about spatio and temporal")
    parser.add_argument('--topk',type=int,default=7)
    parser.add_argument('--epochs', type=int, default=18201)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)
    parser.add_argument('--dataset_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/SHT_I3D_16PATCH.h5")
    parser.add_argument('--model_save_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/model_save/")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Train_new.txt')
    parser.add_argument('--testing_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Test_NEW.txt')
    parser.add_argument('--test_mask_dir',type=str,default='/data/ssy/code/VAD_ST/data/SHT/test_frame_mask/')
    parser.add_argument('--temporal_model_path', type=str, default='')
    parser.add_argument('--classifier_model_path', type=str, default='')


    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    evaluation(args)