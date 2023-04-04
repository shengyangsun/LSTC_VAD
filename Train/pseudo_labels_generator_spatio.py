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
from utils.load_dataset import SH_Train_Origin_Dataset,shanghaitech_test, UCF_test, UBnormal_test
from models.Encoder import Encoder

def generator(args):
    '''
        load spatio model
    '''
    from collections import OrderedDict
    state_dict = torch.load(args.spatio_model_path)
    if args.data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  #delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    spatio_model = Encoder(n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                           d_model=args.d_model, d_inner=args.n_hidden,MHA_layerNorm=args.MHA_layerNorm,
                           FFN_layerNorm=args.FFN_layerNorm, position_dropout=args.position_dropout,
                           weight_init=args.encoder_weight_init, position_encoding=args.position_encoding,
                           CLS_learned=args.CLS_learned, max_position_tokens=args.max_position_tokens,
                           relative_pe=args.relative_position_encoding, window_size=args.window_size, conv_patch=args.conv_patch)
    spatio_model.load_state_dict(new_state_dict, False)
    spatio_model = spatio_model.cuda().eval()
    '''
        load regression model
    '''
    state_dict = torch.load(args.regression_model_path)
    if args.data_parallel == True:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # delete the "module."
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    if args.n_layers == 1:
        regression_model = Classifier(args.d_model)
        regression_model.load_state_dict(new_state_dict, False)
        regression_model = regression_model.cuda().eval()
    else:
        regression_model = Regressor(args.d_model)
        regression_model.load_state_dict(new_state_dict, False)
        regression_model = regression_model.cuda().eval()
    print ("Model load complete.")

    '''
        get pseudo labels
    '''
    pseudo_dict = {}
    dataset_h5 = h5py.File(args.dataset_path, 'r')
    train_lines = open(args.training_txt, 'r').readlines()
    with torch.no_grad():
        for line in tqdm(train_lines):
            if args.dataset == "SHT":
                line_split = line.strip().split(',')
                label, key = int(line_split[1]), line_split[0]
            elif args.dataset == "UCF":
                key = line.strip().split(" ")[0].split('/')[-1].split('.')[0]
            elif args.dataset == "UBnormal":
                key = line.strip().split(",")[0]
            feats = torch.from_numpy(dataset_h5[key + '.npy'][:]).cuda().float()
            feats = spatio_model(feats)
            if args.n_layers == 1:
                logits = regression_model(feats[:, 0, :])[:, 1]
            else:
                logits = regression_model(feats[:, 0, :])
            tensor_zeros = torch.zeros_like(logits)
            logits = torch.where(logits > args.threshold, logits, tensor_zeros)
            scores = logits.cpu().detach().numpy()
            pseudo_dict[key + ".npy"] = scores
    np.save(args.pseudo_labels_path, pseudo_dict)
    print ("spatio pseudo label generation finished.")

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--dataset', type=str, default='SHT')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--segment_len', type=int, default=16)

    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_hidden', type=int, default=3027)
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=256)
    parser.add_argument('--d_v', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--MHA_layerNorm', action="store_true", help = "Run to activate MHA_layerNorm")
    parser.add_argument('--FFN_layerNorm', action="store_true", help = "Run to activate FFN_layerNorm")
    parser.add_argument('--CLS_learned', action="store_true", help = "Run to activate the learning CLS")
    parser.add_argument('--position_encoding', action="store_true", help="Run to activate the position encoding")
    parser.add_argument('--data_parallel', action="store_true", help="data_parallel saved model")
    parser.add_argument('--relative_position_encoding', action="store_true", help="Run to activate the relative position encoding")
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--conv_patch', action="store_true", help="Run to activate the conv patch")
    parser.add_argument('--position_dropout', type=float, default=0.1)
    parser.add_argument('--max_position_tokens', type=int, default=17)
    parser.add_argument('--threshold', type=float, default=0.9)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--dataset_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/SHT_I3D_16PATCH.h5")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Train_new.txt')
    parser.add_argument('--spatio_model_path', type=str, default='/data/ssy/code/VAD_ST/data/SHT/model_save/spatio_model_96.84')
    parser.add_argument('--regression_model_path', type=str,default='/data/ssy/code/VAD_ST/data/SHT/model_save/regression_model_96.84')
    parser.add_argument('--pseudo_labels_path', type=str, default='/data/ssy/code/VAD_ST/data/SHT/clip_pseudo_labels.npy')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    generator(args)