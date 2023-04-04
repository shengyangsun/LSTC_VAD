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
from models.Classifier import Classifier
from utils.load_dataset import SH_Train_Origin_Dataset_MutualTraining,shanghaitech_test, UCF_Train_Origin_Dataset, UCF_test, UBnormal_Train_Origin_Dataset, UBnormal_test
from models.Encoder import Encoder
from collections import OrderedDict

def get_BCE_loss(args, outputs, labs):
    loss = torch.mean(-args.lambda_normal * labs[:, :, 0] * torch.log(1-outputs + (1e-8)) -
                      args.lambda_abnormal * labs[:, :, 1] * torch.log(outputs + (1e-8)))
    return loss

def get_CE_loss(args, outputs, labs):
    loss = F.cross_entropy(outputs, labs)
    return loss

def get_MIL_loss(args,y_pred, part_len):
    topk_pred = torch.max(torch.mean(y_pred.view([args.batch_size * 2, args.part_num, part_len]), dim=-1, keepdim=False), dim=-1, keepdim=False)[0]
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

    if args.dataset != "UCF":
        if args.dataset == "UBnormal":
            test_feats, test_labels, test_annos = UBnormal_test(args.testing_txt, args.test_mask_dir, args.dataset_path)
        else:
            test_feats, test_labels, test_annos = shanghaitech_test(args.testing_txt, args.test_mask_dir, args.dataset_path)

    train_lines = open(args.training_txt, 'r').readlines()
    logger.info("Load dataset complete.")

    spatio_model = Encoder(n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                           d_model=args.d_model, d_inner=args.spatio_n_hidden,
                           MHA_attn_dropout=args.spatio_MHA_attn_dropout, MHA_fc_dropout=args.spatio_MHA_fc_dropout,
                           MHA_layerNorm=args.spatio_MHA_layerNorm, FFN_dropout=args.spatio_FFN_dropout,
                           FFN_layerNorm=args.spatio_FFN_layerNorm, position_dropout=args.position_dropout,
                           weight_init=args.spatio_encoder_weight_init, position_encoding=args.position_encoding,
                           CLS_learned=args.CLS_learned, max_position_tokens=args.max_position_tokens,
                           relative_pe_2D=args.relative_pe_2D, input_layerNorm=args.input_layerNorm)
    regression_model = Regressor(args.d_model, args.regressor_dropout, weight_init=args.regressor_weight_init)

    if args.load_model == True:
        state_dict = torch.load(args.spatio_model_path)
        if args.load_data_parallel == True:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # delete the "module."
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        spatio_model.load_state_dict(new_state_dict, False)
        state_dict = torch.load(args.regression_model_path)
        if args.load_data_parallel == True:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # delete the "module."
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict
        regression_model.load_state_dict(new_state_dict, False)

    if args.data_parallel == True:
        spatio_model = nn.DataParallel(spatio_model)
        regression_model = nn.DataParallel(regression_model)

    spatio_model = spatio_model.cuda().train()
    regression_model = regression_model.cuda().train()

    optimizer = torch.optim.Adagrad([{"params": spatio_model.parameters(), "lr":args.lr_encoder},
                                     {"params": regression_model.parameters(), "lr": args.lr_regressor}],
                                     weight_decay=args.weight_decay)

    spatio_best_test_auc = args.save_threshold
    spatio_best_test_epoch = 0
    spatio_best_train_auc = 0
    spatio_best_train_epoch = 0
    iter_count = 0
    '''
        Train
    '''
    for round_i in range(1):
        if round_i % 2 == 0:
            pseudo_labels_path = args.spatio_pseudo_path
            if args.dataset == "UCF":
                dataset = UCF_Train_Origin_Dataset(part_num=args.part_num, part_len=args.spatio_part_len,
                                                   frames_per_clip=args.segment_len,
                                                   h5_path=args.dataset_path, train_txt=args.training_txt,
                                                   n_patch=args.n_patch,
                                                   sample=args.sample, pseudo_labels_path=pseudo_labels_path)
                test_lines = open(args.testing_txt, 'r').readlines()
            elif args.dataset == "UBnormal":
                dataset = UBnormal_Train_Origin_Dataset(part_num=args.part_num,
                                                                 part_len=args.spatio_part_len,
                                                                 h5_path=args.dataset_path,
                                                                 train_txt=args.training_txt,
                                                                 n_patch=args.n_patch, sample=args.sample,
                                                                 pseudo_labels_path=pseudo_labels_path)
            else:
                dataset = SH_Train_Origin_Dataset_MutualTraining(part_num=args.part_num,
                                                                 part_len=args.spatio_part_len,
                                                                 h5_path=args.dataset_path,
                                                                 train_txt=args.training_txt,
                                                                 n_patch=args.n_patch, sample=args.sample,
                                                                 pseudo_labels_path=pseudo_labels_path)

            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                    worker_init_fn=worker_init, drop_last=True)
            epochs = args.spatio_epochs
            part_len = args.spatio_part_len
        else:
            pseudo_labels_path = args.temporal_pseudo_path+".npy"
            dataset = SH_Train_Origin_Dataset_MutualTraining(part_num=args.temporal_part_num,
                                                                      part_len=args.temporal_part_len,
                                                                      h5_path=args.dataset_path,
                                                                      train_txt=args.training_txt,
                                                                      n_patch=args.n_patch, sample=args.sample,
                                                                      pseudo_labels_path=pseudo_labels_path)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                    worker_init_fn=worker_init, drop_last=True)
            epochs = args.temporal_epochs
            part_len = args.temporal_part_len
        iter_count = 0
        for epoch in range(epochs):
            for norm_feats, norm_labs, abnorm_feats, abnorm_labs in dataloader:
                norm_labs = torch.zeros([args.batch_size, args.part_num, 2], dtype=torch.float32).cuda()
                norm_labs[:, :, 0] += 1
                abnorm_labs = abnorm_labs.cuda().view([args.batch_size, args.part_num, part_len])
                abnorm_labs = torch.mean(abnorm_labs, dim=-1).view([args.batch_size, args.part_num, 1])
                abnorm_labs_tmp = torch.zeros([args.batch_size, args.part_num, 2], dtype=torch.float32).cuda()
                abnorm_labs_tmp[:, :, 1] = abnorm_labs[:, :, 0]
                abnorm_labs_tmp[:, :, 0] = 1 - abnorm_labs_tmp[:, :, 1]
                abnorm_labs = abnorm_labs_tmp
                clip_labs = torch.cat([norm_labs, abnorm_labs], dim=0)

                if round_i % 2 == 0:
                    norm_feats = norm_feats.cuda().float().view(
                        [args.batch_size * args.part_num * part_len, args.n_patch, args.d_model])
                    abnorm_feats = abnorm_feats.cuda().float().view(
                        [args.batch_size * args.part_num * part_len, args.n_patch, args.d_model])
                    feats = torch.cat([norm_feats, abnorm_feats], dim=0)
                    feats = spatio_model(feats)
                    if args.dataset == "UCF":
                        feats = feats[:, 0, :].float().view([args.batch_size * 2, args.part_num * args.spatio_part_len, args.d_model])
                    else:
                        feats = feats[:, 0, :]
                    outputs = regression_model(feats)
                    MIL_loss, err, l1 = get_MIL_loss(args, outputs, part_len)
                    outputs = torch.mean(outputs.view([args.batch_size * 2, args.part_num, part_len]), dim=-1)
                    CE_loss = get_BCE_loss(args, outputs, clip_labs)
                    loss = args.lambda_BCE*CE_loss + MIL_loss
                else:
                    norm_feats = norm_feats.cuda().float().view(
                        [args.batch_size * args.part_num, part_len * args.n_patch, args.d_model])
                    abnorm_feats = abnorm_feats.cuda().float().view(
                        [args.batch_size * args.part_num, part_len * args.n_patch, args.d_model])
                    feats = torch.cat([norm_feats, abnorm_feats], dim=0)
                    feats = temporal_model(feats)
                    feats = feats[:, 0, :].float().view([args.batch_size * 2, args.part_num, args.d_model])
                    outputs = classifier_model(feats)
                    outputs = outputs.view([args.batch_size * 2 * args.part_num, -1])
                    clip_labs = clip_labs.view([args.batch_size * 2 * args.part_num, -1])
                    abnorm_score = outputs[:, 1]
                    CE_loss = get_CE_loss(args, outputs, clip_labs)
                    MIL_loss, err, l1 = get_MIL_loss(args, abnorm_score, 1)
                    loss = args.lambda_MIL * MIL_loss + args.lambda_CE * CE_loss

                optimizer.zero_grad()
                loss.backward()
                if args.clip_grad == True:
                    torch.nn.utils.clip_grad_norm_(spatio_model.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(regression_model.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), 10)
                optimizer.step()
                if round_i % 2 == 0:
                    logger.info('Round {} [{}/{}]: spatio_loss {:.4f}, CE_loss {:.4f}, MIL_loss {:.4f}, err {:.4f}, l1 {:.4f}'.format(
                        round_i, iter_count, epoch, loss, CE_loss, MIL_loss, err, l1))
                else:
                    logger.info('Round {} [{}/{}]: temporal_loss {:.4f}, CE_loss {:.4f}, MIL_loss {:.4f}, err {:.4f}, l1 {:.4f}'.format(
                        round_i, iter_count, epoch, loss, CE_loss, MIL_loss, err, l1))
                iter_count += 1
            dataloader.dataset.shuffle_keys()

            '''
                Evaluation
            '''
            if (epoch % args.inter_epoch == 0) or (epoch == epochs - 1):
                test_scores_list = []
                test_labels_list = []
                train_scores_list = []
                train_labels_list = []

                with torch.no_grad():
                    spatio_model = spatio_model.eval()
                    regression_model = regression_model.eval()

                    if round_i % 2 == 0:
                        if args.dataset == "UCF":
                            max_clips = 21
                            for line in tqdm(test_lines):
                                test_feat, test_anno, n_frames = UCF_test(line, args.dataset_path,
                                                                          args.test_mask_path, args.segment_len)
                                n_clips = n_frames // args.segment_len
                                r = np.linspace(0, n_clips, max_clips + 1, dtype=np.int32)
                                for snippet_i in range(max_clips):
                                    if r[snippet_i] != r[snippet_i + 1]:
                                        feats = np.mean(test_feat[r[snippet_i]:r[snippet_i + 1]], 0)
                                        feats = torch.from_numpy(feats).cuda().float().view(
                                            [-1, args.n_patch, args.d_model])
                                        feats = spatio_model(feats)
                                        logits = regression_model(feats[:, 0, :])
                                        scores = logits.cpu().numpy()
                                        score_tmp = []
                                        for score in scores:
                                            score_tmp.extend(
                                                [score] * args.segment_len * (r[snippet_i + 1] - r[snippet_i]))
                                        test_scores_list.extend(score_tmp)
                                        test_labels_list.extend(test_anno[r[snippet_i] * args.segment_len:r[snippet_i + 1] * args.segment_len].tolist())
                        else:
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
                            if args.dataset != "UBnormal":
                                with h5py.File(args.dataset_path, 'r') as dataset_h5:
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
                    else:
                        for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
                            feats_all = torch.from_numpy(np.array(test_feat)[:, :args.n_patch, :]).cuda().float()
                            n_clips = feats_all.shape[0]
                            total_len = 0
                            n_clip_part = n_clips // args.temporal_part_len
                            if n_clip_part * args.temporal_part_len < n_clips:
                                n_clip_part += 1
                            for clip_part_i in range(n_clip_part):
                                beg = clip_part_i * args.temporal_part_len
                                if clip_part_i == n_clip_part - 1:
                                    end = n_clips
                                else:
                                    end = (clip_part_i + 1) * args.temporal_part_len
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

                        with h5py.File(args.dataset_path, 'r') as dataset_h5:
                            for line in train_lines:
                                line_split = line.strip().split(',')
                                label, key = int(line_split[1]), line_split[0]
                                if label == 1:
                                    gt_path = args.test_mask_dir + key + ".npy"
                                    gt = np.load(gt_path, allow_pickle=True)
                                feats_all = torch.from_numpy(
                                    np.array(dataset_h5[key + '.npy'][:])[:, :args.n_patch, :]).cuda().float()
                                n_clips = feats_all.shape[0]
                                total_len = 0
                                n_clip_part = n_clips // args.temporal_part_len
                                if n_clip_part * args.temporal_part_len < n_clips:
                                    n_clip_part += 1
                                for clip_part_i in range(n_clip_part):
                                    beg = clip_part_i * args.temporal_part_len
                                    if clip_part_i == n_clip_part - 1:
                                        end = n_clips
                                    else:
                                        end = (clip_part_i + 1) * args.temporal_part_len
                                    feats = feats_all[beg:end, :, :].view([1, -1, args.d_model])
                                    feats = temporal_model(feats)
                                    feats = feats[:, 0, :]
                                    logits = classifier_model(feats).view([-1, 2])
                                    logits = logits[:, 1]
                                    score = logits.cpu().numpy()
                                    train_scores_list.extend([score] * (end - beg) * args.segment_len)
                                    if label == 0:
                                        train_labels_list.extend([0] * (end - beg) * args.segment_len)
                                    else:
                                        train_labels_list.extend(
                                            gt[total_len:total_len + (end - beg) * args.segment_len].tolist())
                                    total_len += (end - beg) * args.segment_len


                    auc_test = eval(test_scores_list, test_labels_list, None)
                    if args.dataset == "UCF" or args.dataset == "UBnormal":
                        auc_train = 0
                    else:
                        auc_train = eval(train_scores_list, train_labels_list, None)

                    if round_i % 2 == 0:
                        if auc_train > spatio_best_train_auc:
                            spatio_best_train_auc = auc_train
                            spatio_best_train_epoch = epoch
                            logger.info("saving model......")
                            torch.save(spatio_model.state_dict(), args.model_save_dir + args.saved_prefix+"spatio_model_oneCrop_" + args.type + "_" + str(auc_train))
                            torch.save(regression_model.state_dict(), args.model_save_dir + args.saved_prefix+"regression_model_oneCrop_" + args.type + "_" + str(auc_train))
                            logger.info("save complete.")
                        if auc_test > spatio_best_test_auc:
                            spatio_best_test_auc = auc_test
                            spatio_best_test_epoch = epoch
                        logger.info(
                            'best_train_AUC {} at epoch {} now train_AUC is {}'.format(
                                spatio_best_train_auc, spatio_best_train_epoch, auc_train))
                        logger.info(
                            'best_test_AUC {} at epoch {} now test_AUC is {}'.format(
                                spatio_best_test_auc, spatio_best_test_epoch, auc_test))

                    else:
                        if auc_train > temporal_best_train_auc:
                            temporal_best_train_auc = auc_train
                            temporal_best_train_epoch = epoch
                            logger.info("saving model......")
                            torch.save(temporal_model.state_dict(), args.temporal_model_path)
                            torch.save(classifier_model.state_dict(), args.classifier_model_path)
                            logger.info("save complete.")
                        if auc_test > temporal_best_test_auc:
                            temporal_best_test_auc = auc_test
                            temporal_best_test_epoch = epoch
                        logger.info(
                            'best_train_AUC {} at epoch {} now train_AUC is {}'.format(
                                temporal_best_train_auc, temporal_best_train_epoch, auc_train))
                        logger.info(
                            'best_test_AUC {} at epoch {} now test_AUC is {}'.format(
                                temporal_best_test_auc, temporal_best_test_epoch, auc_test))
                    logger.info(
                        'best_spatio_test_AUC {}'.format(spatio_best_test_auc))
                    logger.info(
                        '======================================================================================')

                spatio_model = spatio_model.train()
                regression_model = regression_model.train()

        if round_i % 2 == 0:
            spatio_model.load_state_dict(torch.load(args.spatio_model_path), False)
            spatio_model = spatio_model.eval()
            regression_model.load_state_dict(torch.load(args.regression_model_path), False)
            regression_model = regression_model.eval()
            pseudo_dict = {}
            train_lines = open(args.training_txt, 'r').readlines()
            with h5py.File(args.dataset_path, 'r') as dataset_h5:
                with torch.no_grad():
                    for line in tqdm(train_lines):
                        line_split = line.strip().split(',')
                        label, key = int(line_split[1]), line_split[0]
                        feats = torch.from_numpy(dataset_h5[key + '.npy'][:]).cuda().float()
                        feats = spatio_model(feats)
                        logits = regression_model(feats[:, 0, :])
                        tensor_zeros = torch.zeros_like(logits)
                        logits = torch.where(logits > args.threshold, logits, tensor_zeros)
                        scores = logits.cpu().numpy()
                        pseudo_dict[key + ".npy"] = scores
            np.save(args.temporal_pseudo_path, pseudo_dict)
            logger.info ("temporal pseudo label generation finished.")
            spatio_model = spatio_model.train()
            regression_model = regression_model.train()
        else:
            temporal_model.load_state_dict(torch.load(args.temporal_model_path_test), False)
            temporal_model = temporal_model.eval()
            classifier_model.load_state_dict(torch.load(args.classifier_model_path_test), False)
            classifier_model = classifier_model.eval()
            pseudo_dict = {}
            train_lines = open(args.training_txt, 'r').readlines()
            with h5py.File(args.dataset_path, 'r') as dataset_h5:
                with torch.no_grad():
                    for line in tqdm(train_lines):
                        line_split = line.strip().split(',')
                        label, key = int(line_split[1]), line_split[0]
                        feats_all = torch.from_numpy(dataset_h5[key + '.npy'][:]).cuda().float()
                        n_clips = feats_all.shape[0]
                        total_len = 0
                        scores_list = []
                        sum_clip = 0
                        n_clip_part = n_clips // args.temporal_part_len
                        if n_clip_part * args.temporal_part_len < n_clips:
                            n_clip_part += 1
                        for clip_part_i in range(n_clip_part):
                            beg = clip_part_i * args.temporal_part_len
                            if clip_part_i == n_clip_part - 1:
                                end = n_clips
                            else:
                                end = (clip_part_i + 1) * args.temporal_part_len
                            sum_clip += end-beg
                            feats = feats_all[beg:end, :, :].view([1, -1, args.d_model])
                            feats = temporal_model(feats)
                            feats = feats[:, 0, :]
                            logits = classifier_model(feats).view([-1, 2])
                            logits = logits[:, 1]
                            tensor_zeros = torch.zeros_like(logits)
                            logits = torch.where(logits > args.threshold, logits, tensor_zeros)
                            score = logits.cpu().numpy()
                            scores_list.extend([score] * (end-beg))
                        pseudo_dict[key + ".npy"] = np.array(scores_list)

            np.save(args.spatio_pseudo_path, pseudo_dict)
            logger.info("spatio pseudo label generation finished.")
            temporal_model = temporal_model.train()
            classifier_model = classifier_model.train()


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Spatio')
    parser.add_argument('--data_crop', type=str, default='oneCrop')
    parser.add_argument('--dataset', type=str, default='MT_SHT')
    parser.add_argument('--type', type=str, default='I3D_RGB')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--rounds_num', type=int, default=500)
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--spatio_part_len', type=int, default=7)
    parser.add_argument('--temporal_part_num', type=int, default=16)
    parser.add_argument('--temporal_part_len', type=int, default=3)

    parser.add_argument('--n_patch', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--spatio_n_hidden', type=int, default=3027)
    parser.add_argument('--d_model', type=int, default=2048)
    parser.add_argument('--d_k', type=int, default=256)
    parser.add_argument('--d_v', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--spatio_MHA_attn_dropout', type=float, default=0.1)
    parser.add_argument('--spatio_MHA_fc_dropout', type=float, default=0.1)
    parser.add_argument('--spatio_FFN_dropout', type=float, default=0.1)
    parser.add_argument('--spatio_MHA_layerNorm', action="store_true", help = "Run to activate MHA_layerNorm")
    parser.add_argument('--spatio_FFN_layerNorm', action="store_true", help = "Run to activate FFN_layerNorm")
    parser.add_argument('--spatio_encoder_weight_init', action="store_true", help = "Run to activate encoder xavier init")
    parser.add_argument('--regressor_weight_init', action="store_true", help = "Run to activate classifier xavier init")
    parser.add_argument('--clip_grad', action="store_true", help = "Run to activate clip grad")
    parser.add_argument('--CLS_learned', action="store_true", help = "Run to activate the learning CLS")
    parser.add_argument('--position_encoding', action="store_true", help="Run to activate the position encoding")
    parser.add_argument('--position_dropout', type=float, default=0.1)
    parser.add_argument('--max_position_tokens', type=int, default=17)
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--relative_pe_2D', action="store_true", help="Run to activate rpe_2D")
    parser.add_argument('--input_layerNorm', action="store_true", help="Run to activate input_layerNorm")
    parser.add_argument('--spatio_epochs', type=int, default=1000000)
    parser.add_argument('--regressor_dropout', type=float, default=0.6)
    parser.add_argument('--lr_regressor', type=float, default=1e-2)
    parser.add_argument('--lambda_BCE', type=float, default=1.0)

    parser.add_argument('--load_model', action="store_true")


    parser.add_argument('--temporal_MHA_attn_dropout', type=float, default=0.2)
    parser.add_argument('--temporal_MHA_fc_dropout', type=float, default=0.2)
    parser.add_argument('--temporal_FFN_dropout', type=float, default=0.1)
    parser.add_argument('--temporal_n_hidden', type=int, default=4096)
    parser.add_argument('--temporal_MHA_layerNorm', action="store_true", help="Run to activate MHA_layerNorm")
    parser.add_argument('--temporal_FFN_layerNorm', action="store_true", help="Run to activate FFN_layerNorm")
    parser.add_argument('--temporal_encoder_weight_init', action="store_true", help="Run to activate encoder xavier init")
    parser.add_argument('--classifier_weight_init', action="store_true", help="Run to activate classifier xavier init")
    parser.add_argument('--relative_position_encoding', action="store_true", help="Run to activate the relative position encoding")
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--conv_patch', action="store_true", help="Run to activate the conv patch")
    parser.add_argument('--temporal_epochs', type=int, default=1000)
    parser.add_argument('--classifier_dropout', type=float, default=0.6)
    parser.add_argument('--lr_classifier', type=float, default=1e-2)
    parser.add_argument('--lambda_MIL', type=float, default=1.0)
    parser.add_argument('--lambda_CE', type=float, default=0.8)

    parser.add_argument('--lambda_normal', type=float, default=0.2)
    parser.add_argument('--lambda_abnormal', type=float, default=2.0)

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
    parser.add_argument('--model_save_dir', type=str, default="/data/ssy/code/VAD_ST/data/SHT/model_save/")
    parser.add_argument('--version', type=str, default="1.0")
    parser.add_argument('--training_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Train_new.txt')
    parser.add_argument('--testing_txt',type=str,default='/data/ssy/code/VAD_ST/data/SHT/SH_Test_NEW.txt')
    parser.add_argument('--test_mask_dir',type=str,default='/data/ssy/code/VAD_ST/data/SHT/test_frame_mask/')
    parser.add_argument('--test_mask_path', type=str, default='/data/ssy/code/VAD_ST/data/UCF_Crime/UCF_Crime_gt.h5')
    parser.add_argument('--inter_epoch', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.85)

    parser.add_argument('--load_data_parallel', action="store_true")
    parser.add_argument('--spatio_pseudo_path', type=str, default="/data/ssy/code/VAD_ST/data/SHT/spatio_model_pseudo_labels3")
    parser.add_argument('--spatio_model_path', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/spatio_model3")
    parser.add_argument('--regression_model_path', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/regression_model3")
    parser.add_argument('--temporal_pseudo_path', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/temporal_model_pseudo_labels3")
    parser.add_argument('--temporal_model_path', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/temporal_model3")
    parser.add_argument('--classifier_model_path', type=str,
                        default="/data/ssy/code/VAD_ST/data/SHT/classifier_model3")
    parser.add_argument('--saved_prefix', type=str, default='')

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args=parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    train(args)