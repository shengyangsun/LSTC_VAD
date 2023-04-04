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
from utils.load_dataset import UCF_Train_Origin_Dataset, UCF_test
from models.Encoder import Encoder


def evaluation(args):
    '''
        load dataset
    '''
    test_lines = open(args.testing_txt, 'r').readlines()
    print ("Dataset load complete.")
    print ('Dataset: UCF')
    temporal_model = Encoder(n_layers=args.temporal_n_layers, n_head=args.temporal_n_head, d_k=args.temporal_d_k,
                             d_v=args.temporal_d_v, d_model=args.d_model, d_inner=args.temporal_n_hidden,
                             MHA_layerNorm=args.temporal_MHA_layerNorm, FFN_layerNorm=args.temporal_FFN_layerNorm,
                             relative_pe=args.relative_position_encoding, window_size=args.window_size, window_depth=args.part_len)
    temporal_model.load_state_dict(torch.load(args.temporal_model_path), False)
    temporal_model = temporal_model.cuda().eval()
    '''
        load classifier model
    '''
    classifier_model = Classifier(args.d_model)
    classifier_model.load_state_dict(torch.load(args.classifier_model_path), False)
    classifier_model = classifier_model.cuda().eval()
    args.part_len = 2

    max_clips  = 32
    temporal_test_scores_list = []
    temporal_test_labels_list = []
    with torch.no_grad():
        for line in test_lines:
            test_feat, test_anno, n_frames, video_name = UCF_test(line, args.dataset_path, args.test_mask_path,
                                                                  args.segment_len, return_name=True)
            feats_all = torch.from_numpy(np.array(test_feat)).cuda().float()
            n_clips = n_frames // args.segment_len
            current_clips = max_clips
            r = np.linspace(0, n_clips, current_clips + 1, dtype=np.int32)
            n_clip_part = current_clips // args.part_len
            if n_clip_part * args.part_len < current_clips:
                n_clip_part += 1
            for clip_part_i in range(n_clip_part):
                beg = clip_part_i * args.part_len
                if clip_part_i == n_clip_part - 1:
                    end = current_clips
                else:
                    end = (clip_part_i + 1) * args.part_len
                if end - beg < args.part_len:
                    beg = end - args.part_len
                for r_i in range(beg, end):
                    if r[r_i] == r[r_i + 1]:
                        feature = feats_all[r[r_i]].view([-1, args.n_patch, args.d_model])
                    else:
                        feature = feats_all[r[r_i]:r[r_i + 1]].view([-1, args.n_patch, args.d_model])
                        feature = torch.mean(feature, dim=0).view([-1, args.n_patch, args.d_model])
                    if r_i == beg:
                        feat_cat = feature
                    else:
                        feat_cat = torch.cat([feat_cat, feature], dim=0)
                feats = feat_cat.view([1, -1, args.d_model])
                feats = F.normalize(feats, p=2, dim=-1)
                feats = temporal_model(feats)
                feats = feats[:, 0, :]
                logits = classifier_model(feats).view([-1, 2])
                logits = logits[:, 1]
                score = logits.cpu().numpy()
                temporal_test_scores_list.extend([score] * (r[end] - r[beg]) * args.segment_len)
                temporal_test_labels_list.extend(
                    test_anno[r[beg] * args.segment_len:r[end] * args.segment_len].tolist())

        auc_test = eval(temporal_test_scores_list, temporal_test_labels_list, None)
        print("auc=", auc_test)

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--part_len', type=int, default=2)

    parser.add_argument('--n_patch', type=int, default=9)
    parser.add_argument('--d_model', type=int, default=2048)

    parser.add_argument('--temporal_n_head', type=int, default=8)
    parser.add_argument('--temporal_n_hidden', type=int, default=4096)
    parser.add_argument('--temporal_d_k', type=int, default=256)
    parser.add_argument('--temporal_d_v', type=int, default=256)
    parser.add_argument('--temporal_n_layers', type=int, default=3)
    parser.add_argument('--temporal_MHA_layerNorm', action="store_true", help="Run to activate MHA_layerNorm of temporal")
    parser.add_argument('--temporal_FFN_layerNorm', action="store_true", help="Run to activate FFN_layerNorm of temporal")
    parser.add_argument('--relative_position_encoding', action="store_true",
                        help="Run to activate the relative position encoding")
    parser.add_argument('--window_size', type=int, default=4)

    parser.add_argument('--generate_clip_labels', action="store_true", help="Run to generate clip labels about spatio and temporal")
    parser.add_argument('--topk',type=int,default=7)
    parser.add_argument('--epochs', type=int, default=18201)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)
    parser.add_argument('--dataset_path', type=str, default="/ssd/ssy/UCF/UCF_I3D_9PATCH_32_norm.h5")
    parser.add_argument('--model_save_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/model_save/")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--testing_txt',type=str,default='/data/ssy/code/VAD_ST/data/UCF_Crime/Test_Annotation.txt')
    parser.add_argument('--test_mask_path', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/UCF_Crime_gt.h5')
    parser.add_argument('--temporal_model_path', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/model_save/temporal_model_oneCrop_I3D_RGB_0.8570')
    parser.add_argument('--classifier_model_path', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/model_save/classifier_model_oneCrop_I3D_RGB_0.8570')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    evaluation(args)