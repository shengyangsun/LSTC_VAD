import os
import sys
sys.path.append('..')
import argparse
import time
import random
import math
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from utils.utils import *
from utils.eval_utils import eval
from tqdm import tqdm
from models.Regressor import Regressor
from utils.load_dataset import SH_Train_Origin_Dataset,shanghaitech_test
from models.Encoder import Encoder

def get_MIL_loss(args,y_pred):
    topk_pred = torch.max(torch.mean(y_pred.view([args.batch_size * 2, args.part_num, args.part_len]), dim=-1, keepdim=False), dim=-1, keepdim=False)[0]
    nor_max=topk_pred[:args.batch_size]
    abn_max=topk_pred[args.batch_size:]
    err=0
    for i in range(args.batch_size):
        err+=torch.sum(F.relu(1-abn_max+nor_max[i]))
    err=err/(args.batch_size)**2
    abn_pred=y_pred[args.batch_size:]
    spar_l1=torch.mean(abn_pred)
    loss=err+args.lambda_1*spar_l1
    return loss,err,spar_l1


def train(args):
    def worker_init(worked_id):
        np.random.seed(args.seed + worked_id)
        random.seed(args.seed + worked_id)

    logger = log_setting(args)
    dataset = SH_Train_Origin_Dataset(part_num=args.part_num, part_len=args.part_len, h5_path=args.dataset_path,
                                      train_txt=args.training_txt, n_patch=args.n_patch,
                                      sample=args.sample, pseudo_labels_path=None)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init, drop_last=True)
    test_feats, test_labels, test_annos = shanghaitech_test(args.testing_txt, args.test_mask_dir, args.dataset_path)


    train_lines = open(args.training_txt, 'r').readlines()
    logger.info("Load dataset complete.")

    spatio_model = Encoder(n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                           d_model=args.d_model, d_inner=args.n_hidden,
                           MHA_attn_dropout=args.MHA_attn_dropout, MHA_fc_dropout=args.MHA_fc_dropout,
                           MHA_layerNorm=args.MHA_layerNorm,FFN_dropout=args.FFN_dropout,
                           FFN_layerNorm=args.FFN_layerNorm, position_dropout=args.position_dropout,
                           weight_init=args.encoder_weight_init, position_encoding=args.position_encoding,
                           CLS_learned=args.CLS_learned, max_position_tokens=args.max_position_tokens,
                           relative_pe_2D=args.relative_pe_2D, input_layerNorm=args.input_layerNorm)

    regression_model = Regressor(args.d_model, args.regressor_dropout, weight_init=args.regressor_weight_init)

    if args.load_model == True:
        state_dict = torch.load(args.load_spatio_model_path)
        spatio_model.load_state_dict(state_dict, False)
        state_dict = torch.load(args.load_classifier_model_path)
        regression_model.load_state_dict(state_dict, False)

    if args.data_parallel == True:
        spatio_model = nn.DataParallel(spatio_model)
        regression_model = nn.DataParallel(regression_model)

    spatio_model = spatio_model.cuda().train()
    regression_model = regression_model.cuda().train()

    optimizer = torch.optim.Adagrad([{"params": spatio_model.parameters(), "lr":args.lr_encoder},
                                     {"params": regression_model.parameters(), "lr": args.lr_regressor}],
                                     weight_decay=args.weight_decay)

    best_test_auc = 0
    best_test_epoch = 0
    best_train_auc = 0
    best_train_epoch = 0
    iter_count = 0
    '''
        Train
    '''
    for epoch in range(args.epochs):
        for norm_feats, norm_labs, abnorm_feats, abnorm_labs in dataloader:
            norm_feats = norm_feats.cuda().float().view(
                [args.batch_size * args.part_num * args.part_len, args.n_patch, args.d_model])
            abnorm_feats = abnorm_feats.cuda().float().view(
                [args.batch_size * args.part_num * args.part_len, args.n_patch, args.d_model])
            feats = torch.cat([norm_feats, abnorm_feats], dim=0)

            feats = spatio_model(feats)
            feats = feats[:, 0, :].float().view([args.batch_size * 2, args.part_num * args.part_len, args.d_model])

            outputs = regression_model(feats)
            outputs = outputs.view([args.batch_size * 2, args.part_num * args.part_len, -1])
            loss, err, l1 = get_MIL_loss(args, outputs)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad == True:
                torch.nn.utils.clip_grad_norm_(spatio_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(regression_model.parameters(), 10)

            optimizer.step()

            logger.info('[{}/{}]: loss {:.4f}, err {:.4f}, l1 {:.4f}'.format(
                iter_count, epoch, loss, err, l1))
            iter_count += 1

        dataloader.dataset.shuffle_keys()

        '''
            Evaluation
        '''
        if epoch % args.inter_epoch == 0:
            test_scores_list = []
            test_labels_list = []
            train_scores_list = []
            train_labels_list = []

            with torch.no_grad():
                spatio_model = spatio_model.eval()
                regression_model = regression_model.eval()

                '''
                Test dataset
                '''
                for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
                    feats = torch.from_numpy(np.array(test_feat)[:, :args.n_patch, :]).cuda().float()
                    feats = spatio_model(feats)
                    logits = regression_model(feats[:, 0, :])
                    scores = logits.cpu().numpy()

                    score_tmp = []
                    for score in scores:
                        score_tmp.extend([score] * args.segment_len)
                    test_labels_list.extend(test_anno[:len(score_tmp)].tolist())
                    test_scores_list.extend(score_tmp)

                '''
                    Train dataset
                '''
                with h5py.File(args.train_dataset, 'r') as dataset_h5:
                    for line in train_lines:
                        line_split = line.strip().split(',')
                        label, key = int(line_split[1]), line_split[0]
                        if label == 1:
                            gt_path = args.test_mask_dir + key + ".npy"
                            gt = np.load(gt_path, allow_pickle=True)
                        feats = torch.from_numpy(
                            np.array(dataset_h5[key + '.npy'][:])[:, :args.n_patch, :]).cuda().float()
                        feats = spatio_model(feats)
                        logits = regression_model(feats[:, 0, :])
                        scores = logits.cpu().numpy()

                        score_tmp = []
                        for score in scores:
                            score_tmp.extend([score] * args.segment_len)
                        if label == 0:
                            train_labels_list.extend([0] * len(score_tmp))
                        else:
                            train_labels_list.extend(gt[:len(score_tmp)].tolist())
                        train_scores_list.extend(score_tmp)


            auc_test = eval(test_scores_list, test_labels_list, None)
            auc_train = eval(train_scores_list, train_labels_list, None)

            if auc_test > best_test_auc:
                best_test_auc = auc_test
                best_test_epoch = epoch
            if auc_train > best_train_auc:
                best_train_auc = auc_train
                best_train_epoch = epoch
                '''
                    save model
                '''
                if auc_train > args.save_threshold:
                    logger.info("saving model......")
                    torch.save(spatio_model.state_dict(),
                               args.model_save_dir + args.saved_prefix + "spatio_model_oneCrop_" + args.type + "_" + str(
                                   auc_train))
                    torch.save(regression_model.state_dict(),
                               args.model_save_dir + args.saved_prefix + "regression_model_oneCrop_" + args.type + "_" + str(
                                   auc_train))
                    logger.info("save complete.")

            logger.info('best_test_AUC {} at epoch {} now test_AUC is {} \nbest_train_AUC {} at epoch {} now train_AUC is {}'.format(
                    best_test_auc, best_test_epoch, auc_test, best_train_auc, best_train_epoch, auc_train))
            logger.info('======================================================================================')

            spatio_model = spatio_model.train()
            regression_model = regression_model.train()


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Spatio')
    parser.add_argument('--data_crop', type=str, default='oneCrop')
    parser.add_argument('--dataset', type=str, default='SHT')
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--part_len', type=int, default=7)

    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_hidden', type=int, default=3027)
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=256)
    parser.add_argument('--d_v', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--MHA_attn_dropout', type=float, default=0.1)
    parser.add_argument('--MHA_fc_dropout', type=float, default=0.1)
    parser.add_argument('--FFN_dropout', type=float, default=0.1)
    parser.add_argument('--MHA_layerNorm', action="store_true", help = "Run to activate MHA_layerNorm")
    parser.add_argument('--FFN_layerNorm', action="store_true", help = "Run to activate FFN_layerNorm")
    parser.add_argument('--encoder_weight_init', action="store_true", help = "Run to activate encoder xavier init")
    parser.add_argument('--regressor_weight_init', action="store_true", help = "Run to activate classifier xavier init")
    parser.add_argument('--clip_grad', action="store_true", help = "Run to activate clip grad")
    parser.add_argument('--CLS_learned', action="store_true", help = "Run to activate the learning CLS")
    parser.add_argument('--position_encoding', action="store_true", help="Run to activate the position encoding")
    parser.add_argument('--position_dropout', type=float, default=0.1)
    parser.add_argument('--max_position_tokens', type=int, default=17)
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--relative_pe_2D', action="store_true", help="Run to activate rpe_2D")
    parser.add_argument('--input_layerNorm', action="store_true", help="Run to activate input_layerNorm")

    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--load_spatio_model_path', type=str, default="null path")
    parser.add_argument('--load_classifier_model_path', type=str, default="null path")


    parser.add_argument('--regressor_dropout', type=float, default=0.6)
    parser.add_argument('--lr_regressor', type=float, default=1e-2)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_parallel', action="store_true", help="Run to activate data parallel")
    parser.add_argument('--save_threshold', type=float, default=0.9685)
    parser.add_argument('--topk',type=int,default=7)
    parser.add_argument('--epochs', type=int, default=18201)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)
    parser.add_argument('--dataset_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/SHT_I3D_16PATCH.h5")
    parser.add_argument('--train_dataset', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/SHT_I3D_16PATCH.h5")
    parser.add_argument('--model_save_dir', type=str, default="/data/ssy/code/VAD_ST/data/SHT/model_save/")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Train_new.txt')
    parser.add_argument('--testing_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Test_NEW.txt')
    parser.add_argument('--test_mask_dir',type=str,default='/data/ssy/code/VAD_ST/data/SHT/test_frame_mask/')
    parser.add_argument('--saved_prefix', type=str, default='')
    parser.add_argument('--inter_epoch', type=int, default=10)


    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    train(args)