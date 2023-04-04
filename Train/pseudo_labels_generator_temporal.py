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
from utils.load_dataset import SH_Train_Origin_Dataset,shanghaitech_test, UCF_test, UCF_train, UBnormal_test
from models.Encoder import Encoder

def generator(args):
    '''
        load spatio model
    '''
    from collections import OrderedDict
    state_dict = torch.load(args.temporal_model_path)
    if args.data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  #delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    temporal_model = Encoder(n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                           d_model=args.d_model, d_inner=args.n_hidden,MHA_layerNorm=args.MHA_layerNorm,
                           FFN_layerNorm=args.FFN_layerNorm, position_dropout=args.position_dropout,
                           weight_init=args.encoder_weight_init, position_encoding=args.position_encoding,
                           CLS_learned=args.CLS_learned, max_position_tokens=args.max_position_tokens,
                           relative_pe=args.relative_position_encoding, window_size=args.window_size, conv_patch=args.conv_patch,
                           window_depth=args.part_len)
    temporal_model.load_state_dict(new_state_dict, False)
    temporal_model = temporal_model.cuda().eval()
    '''
        load classifier model
    '''
    state_dict = torch.load(args.classifier_model_path)
    if args.data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    if args.n_layers == 1:
        classifier_model = Classifier(args.d_model)
        classifier_model.load_state_dict(new_state_dict, False)
        classifier_model = classifier_model.cuda().eval()
    else:
        classifier_model = Classifier(args.d_model)
        classifier_model.load_state_dict(new_state_dict, False)
        classifier_model = classifier_model.cuda().eval()
    print ("Model load complete.")

    pseudo_dict = {}
    if args.dataset == "UCF":
        temporal_test_scores_list = []
        temporal_test_labels_list = []
        train_lines = open(args.training_txt, 'r').readlines()
        max_clips = 32
        with torch.no_grad():
            for line in tqdm(train_lines):
                test_feat, n_frames, key = UCF_train(line, args.dataset_path, args.segment_len, return_name=True)
                feats_all = torch.from_numpy(np.array(test_feat)).cuda().float()
                scores = []
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
                    feats = temporal_model(feats)
                    feats = feats[:, 0, :]
                    logits = classifier_model(feats).view([-1, 2])
                    logits = logits[:, 1]
                    tensor_zeros = torch.zeros_like(logits)
                    logits = torch.where(logits > args.threshold, logits, tensor_zeros)
                    score = logits.cpu().numpy()
                    scores.extend([score] * (end - beg))
                pseudo_dict[key + ".npy"] = np.array(scores)
        np.save(args.pseudo_labels_path, pseudo_dict)
    else:
        train_lines = open(args.training_txt, 'r').readlines()
        with h5py.File(args.dataset_path, 'r') as dataset_h5:
            with torch.no_grad():
                for line in tqdm(train_lines):
                    line_split = line.strip().split(',')
                    if args.dataset == "SHT":
                        label, key = int(line_split[1]), line_split[0]
                    elif args.dataset == "UBnormal":
                        key = line_split[0]
                    feats_all = torch.from_numpy(dataset_h5[key + '.npy'][:]).cuda().float()
                    n_clips = feats_all.shape[0]
                    total_len = 0
                    scores_list = []
                    sum_clip = 0
                    n_clip_part = n_clips // args.part_len
                    if n_clip_part * args.part_len < n_clips:
                        n_clip_part += 1
                    for clip_part_i in range(n_clip_part):
                        beg = clip_part_i * args.part_len
                        if clip_part_i == n_clip_part - 1:
                            end = n_clips
                        else:
                            end = (clip_part_i + 1) * args.part_len
                        sum_clip += end - beg
                        feats = feats_all[beg:end, :, :].view([1, -1, args.d_model])
                        feats = temporal_model(feats)
                        feats = feats[:, 0, :]
                        logits = classifier_model(feats).view([-1, 2])
                        logits = logits[:, 1]
                        tensor_zeros = torch.zeros_like(logits)
                        logits = torch.where(logits > args.threshold, logits, tensor_zeros)
                        score = logits.cpu().numpy()
                        scores_list.extend([score] * (end - beg))
                    pseudo_dict[key + ".npy"] = np.array(scores_list)

    np.save(args.pseudo_labels_path, pseudo_dict)
    print ("temporal pseudo label generation finished.")

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--dataset', type=str, default='SHT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--segment_len', type=int, default=16)

    parser.add_argument('--part_len', type=int, default=3)
    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_hidden', type=int, default=3027)
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=256)
    parser.add_argument('--d_v', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--MHA_dropout', type=float, default=0.1)
    parser.add_argument('--FFN_dropout', type=float, default=0.1)
    parser.add_argument('--MHA_layerNorm', action="store_true", help = "Run to activate MHA_layerNorm")
    parser.add_argument('--FFN_layerNorm', action="store_true", help = "Run to activate FFN_layerNorm")
    parser.add_argument('--clip_grad', action="store_true", help = "Run to activate clip grad")
    parser.add_argument('--CLS_learned', action="store_true", help = "Run to activate the learning CLS")
    parser.add_argument('--position_encoding', action="store_true", help="Run to activate the position encoding")
    parser.add_argument('--data_parallel', action="store_true", help="data_parallel saved model")
    parser.add_argument('--relative_position_encoding', action="store_true", help="Run to activate the relative position encoding")
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--conv_patch', action="store_true", help="Run to activate the conv patch")
    parser.add_argument('--position_dropout', type=float, default=0.1)
    parser.add_argument('--max_position_tokens', type=int, default=17)
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.9)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/SHT_I3D_16PATCH.h5")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Train_new.txt')
    parser.add_argument('--testing_txt', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/Test_Annotation.txt')
    parser.add_argument('--temporal_model_path', type=str, default='/data/ssy/code/VAD_ST/data/SHT/model_save/spatio_model_96.84')
    parser.add_argument('--classifier_model_path', type=str,default='/data/ssy/code/VAD_ST/data/SHT/model_save/classifier_model_96.84')
    parser.add_argument('--pseudo_labels_path', type=str, default='/data/ssy/code/VAD_ST/data/SHT/clip_pseudo_labels.npy')
    parser.add_argument('--test_mask_path', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/UCF_Crime_gt.h5')
    parser.add_argument('--test_mask_dir', type=str, default='/data/ssy/code/VAD_ST/data/SHT/test_frame_mask/')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    generator(args)