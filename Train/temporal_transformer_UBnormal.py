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
from utils.utils import *
from utils.eval_utils import eval
from tqdm import tqdm
from models.Classifier import Classifier
from utils.load_dataset import UBnormal_Train_Origin_Dataset, UBnormal_test
from models.Encoder import Encoder


def get_CE_loss(args, outputs, labs):
    loss = F.cross_entropy(outputs, labs)
    return loss

def get_MIL_loss(args, y_pred):
    topk_pred = torch.max(y_pred.view([args.batch_size * 2, args.part_num]), dim=-1, keepdim=False)[0]
    nor_max = topk_pred[:args.batch_size]
    abn_max = topk_pred[args.batch_size:]
    err = 0
    for i in range(args.batch_size):
        err += torch.sum(F.relu(1 - abn_max + nor_max[i]))
    err = err / (args.batch_size) ** 2
    abn_pred = y_pred[args.batch_size:]
    spar_l1 = torch.mean(abn_pred)
    loss = err + args.lambda_1 * spar_l1
    return loss, err, spar_l1


def train(args):
    def worker_init(worked_id):
        np.random.seed(args.seed + worked_id)
        random.seed(args.seed + worked_id)

    logger = log_setting(args)
    dataset = UBnormal_Train_Origin_Dataset(part_num=args.part_num, part_len=args.part_len, h5_path=args.dataset_path,
                                            train_txt=args.training_txt, n_patch=args.n_patch,
                                            sample=args.sample, pseudo_labels_path=args.pseudo_labels_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, worker_init_fn=worker_init,
                            drop_last=True)
    test_feats, test_labels, test_annos = UBnormal_test(args.testing_txt, args.test_mask_dir, args.dataset_path)

    dataset_h5 = h5py.File(args.dataset_path, 'r')
    train_lines = open(args.training_txt, 'r').readlines()
    logger.info("Load dataset complete.")

    temporal_model = Encoder(n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                             d_model=args.d_model, d_inner=args.n_hidden,
                             MHA_attn_dropout=args.MHA_attn_dropout, MHA_fc_dropout=args.MHA_fc_dropout,
                             MHA_layerNorm=args.MHA_layerNorm, FFN_dropout=args.FFN_dropout,
                             FFN_layerNorm=args.FFN_layerNorm, position_dropout=args.position_dropout,
                             weight_init=args.encoder_weight_init, position_encoding=args.position_encoding,
                             CLS_learned=args.CLS_learned, max_position_tokens=args.max_position_tokens,
                             relative_pe=args.relative_position_encoding, window_size=args.window_size,
                             window_depth=args.part_len, conv_patch=args.conv_patch)

    classifier_model = Classifier(args.d_model, args.classifier_dropout, weight_init=args.classifier_weight_init)

    if args.load_model == True:
        state_dict = torch.load(args.load_temporal_model_path)
        temporal_model.load_state_dict(state_dict, False)
        state_dict = torch.load(args.load_classifier_model_path)
        classifier_model.load_state_dict(state_dict, False)

    if args.data_parallel == True:
        temporal_model = nn.DataParallel(temporal_model)
        classifier_model = nn.DataParallel(classifier_model)

    temporal_model = temporal_model.cuda().train()
    classifier_model = classifier_model.cuda().train()

    optimizer = torch.optim.Adagrad([{"params": temporal_model.parameters(), "lr": args.lr_encoder},
                                     {"params": classifier_model.parameters(), "lr": args.lr_classifier}],
                                    weight_decay=args.weight_decay)

    max_grad_temporal = 0
    max_grad_class = 0

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
            '''
                process pseudo labels
            '''
            if args.temporal_only == False:
                norm_labs = torch.zeros([args.batch_size, args.part_num, 2], dtype=torch.float32).cuda()
                norm_labs[:, :, 0] += 1
                abnorm_labs = abnorm_labs.cuda().view([args.batch_size, args.part_num, args.part_len])
                abnorm_labs = torch.mean(abnorm_labs, dim=-1).view([args.batch_size, args.part_num, 1])
                abnorm_labs_tmp = torch.zeros([args.batch_size, args.part_num, 2], dtype=torch.float32).cuda()
                abnorm_labs_tmp[:, :, 1] = abnorm_labs[:, :, 0]
                abnorm_labs_tmp[:, :, 0] = 1 - abnorm_labs_tmp[:, :, 1]
                abnorm_labs = abnorm_labs_tmp
                clip_labs = torch.cat([norm_labs, abnorm_labs], dim=0)

            norm_feats = norm_feats.cuda().float().view(
                [args.batch_size * args.part_num, args.part_len * args.n_patch, args.d_model])
            abnorm_feats = abnorm_feats.cuda().float().view(
                [args.batch_size * args.part_num, args.part_len * args.n_patch, args.d_model])

            feats = torch.cat([norm_feats, abnorm_feats], dim=0)

            feats = temporal_model(feats)
            feats = feats[:, 0, :].float().view([args.batch_size * 2, args.part_num, args.d_model])

            outputs = classifier_model(feats)
            outputs = outputs.view([args.batch_size * 2 * args.part_num, -1])
            abnorm_score = outputs[:, 1]

            if args.temporal_only == False:
                clip_labs = clip_labs.view([args.batch_size * 2 * args.part_num, -1])
                CE_loss = get_CE_loss(args, outputs, clip_labs)
            else:
                CE_loss = 0
            MIL_loss, err, l1 = get_MIL_loss(args, abnorm_score)
            loss = args.lambda_MIL * MIL_loss + args.lambda_CE * CE_loss

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad == True:
                torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 10)

            optimizer.step()

            logger.info(
                '[{}/{}]: loss {:.4f}, MIL_loss {:.4f}, CE_loss {:.4f} MIL_l1 {:.4f}'.format(iter_count, epoch, loss,
                                                                                             MIL_loss, CE_loss, l1))
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
                temporal_model = temporal_model.eval()
                classifier_model = classifier_model.eval()
                '''
                    Test data
                '''
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
                        test_labels_list.extend(
                            test_anno[total_len:total_len + (end - beg) * args.segment_len].tolist())
                        total_len += (end - beg) * args.segment_len
                '''
                    Train data
                '''
                for line in train_lines:
                    line_split = line.strip().split(',')
                    label, key = int(line_split[1]), line_split[0]
                    if label == 1:
                        gt_path = args.test_mask_dir + key + ".npy"
                        gt = np.load(gt_path, allow_pickle=True)
                    feats_all = torch.from_numpy(np.array(dataset_h5[key + '.npy'][:])[:, :args.n_patch, :]).cuda().float()
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
                        if end-beg < args.part_len:
                            feats = feats_all[end-args.part_len:end, :, :].view([1, -1, args.d_model])
                        else:
                            feats = feats_all[beg:end, :, :].view([1, -1, args.d_model])
                        feats = temporal_model(feats)
                        feats = feats[:, 0, :]
                        logits = classifier_model(feats).view([-1, 2])
                        logits = logits[:, 1]
                        score = logits.cpu().numpy()

                        train_scores_list.extend([score] * (end-beg) * args.segment_len)
                        if label == 0:
                            train_labels_list.extend([0] * (end-beg) * args.segment_len)
                        else:
                            train_labels_list.extend(gt[total_len:total_len + (end-beg) * args.segment_len].tolist())
                        total_len += (end - beg) * args.segment_len

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
                    torch.save(temporal_model.state_dict(),
                               args.model_save_dir + "temporal_model_oneCrop_" + args.type + "_" + str(auc_train))
                    torch.save(classifier_model.state_dict(),
                               args.model_save_dir + "classifier_model_oneCrop_" + args.type + "_" + str(auc_train))
                    logger.info("save complete.")

            logger.info('best_test_AUC {} at epoch {} now test_AUC is {} \nbest_train_AUC {} at epoch {} now train_AUC is {}'.format(
                    best_test_auc, best_test_epoch, auc_test, best_train_auc, best_train_epoch, auc_train))
            logger.info(
                'best_test_AUC {} at epoch {} now test_AUC is {}'.format(best_test_auc, best_test_epoch, auc_test))
            logger.info('======================================================================================')

            temporal_model = temporal_model.train()
            classifier_model = classifier_model.train()


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Temporal')
    parser.add_argument('--data_crop', type=str, default='oneCrop')
    parser.add_argument('--dataset', type=str, default='UBnormal')
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--part_len', type=int, default=5)

    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_hidden', type=int, default=4096)
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=256)
    parser.add_argument('--d_v', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--MHA_attn_dropout', type=float, default=0.2)
    parser.add_argument('--MHA_fc_dropout', type=float, default=0.2)
    parser.add_argument('--FFN_dropout', type=float, default=0.1)
    parser.add_argument('--MHA_layerNorm', action="store_true", help="Run to activate MHA_layerNorm")
    parser.add_argument('--FFN_layerNorm', action="store_true", help="Run to activate FFN_layerNorm")
    parser.add_argument('--encoder_weight_init', action="store_true", help="Run to activate encoder xavier init")
    parser.add_argument('--classifier_weight_init', action="store_true", help="Run to activate classifier xavier init")
    parser.add_argument('--clip_grad', action="store_true", help="Run to activate clip grad")
    parser.add_argument('--CLS_learned', action="store_true", help="Run to activate the learning CLS")
    parser.add_argument('--position_encoding', action="store_true", help="Run to activate the position encoding")
    parser.add_argument('--relative_position_encoding', action="store_true",
                        help="Run to activate the relative position encoding")
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--max_clips', type=int, default=32)
    parser.add_argument('--conv_patch', action="store_true", help="Run to activate the conv patch")
    parser.add_argument('--position_dropout', type=float, default=0.1)
    parser.add_argument('--max_position_tokens', type=int, default=17)
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--data_parallel', action="store_true", help="Run to activate data parallel")

    parser.add_argument('--load_model', action="store_true")

    parser.add_argument('--temporal_only', action="store_true")

    parser.add_argument('--classifier_dropout', type=float, default=0.6)
    parser.add_argument('--lr_classifier', type=float, default=1e-2)

    parser.add_argument('--inter_epoch', type=int, default=10)
    parser.add_argument('--save_threshold', type=float, default=0.9713)
    parser.add_argument('--epochs', type=int, default=18201)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_MIL', type=float, default=1.0)
    parser.add_argument('--lambda_CE', type=float, default=0.8)

    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--model_save_dir', type=str, default="")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt', type=str, default='')
    parser.add_argument('--testing_txt', type=str, default='')
    parser.add_argument('--test_mask_dir', type=str, default='')
    parser.add_argument('--pseudo_labels_path', type=str,
                        default='')

    parser.add_argument('--load_temporal_model_path', type=str, default="")
    parser.add_argument('--load_classifier_model_path', type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    train(args)