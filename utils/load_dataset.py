import torch
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import random
import h5py
from utils.utils import decode_imgs

class SH_Train_Origin_Dataset(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, n_patch, sample, pseudo_labels_path=None):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch
        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []
        lab_dict = {'Normal': 0, 'Abnormal': 1}
        h5 = h5py.File(self.h5_path, 'r')
        lines = open(self.train_txt, 'r').readlines()
        self.norm_keys = []
        self.abnorm_keys = []
        for line in lines:
            line_split = line.strip().split(',')
            label = int(line_split[-1])
            key = line_split[0]
            if label == 0:
                self.norm_feats.append(h5[key + '.npy'][:])
                if self.pseudo_labels != None:
                    self.norm_keys.append(key + '.npy')
            else:
                self.abnorm_feats.append(h5[key + '.npy'][:])
                if self.pseudo_labels != None:
                    self.abnorm_keys.append(key + '.npy')
    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat, labs, vid_type='Normal'):
        #feat = np.array(feat)
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)

        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]

        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])
        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter]]
        else:
            norm_labs = abnorm_labs = None
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter], abnorm_labs, vid_type='Abnormal')

        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:
            return torch.from_numpy(norm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(abnorm_labs).float()

def shanghaitech_test(txt_path, mask_dir, h5_file, return_names=False):
    lines = open(txt_path,'r').readlines()
    annos = []
    labels = []
    names = []
    h5 = h5py.File(h5_file, 'r')
    output_feats = []
    for line in lines:
        line_split = line.strip().split(',')
        key = line_split[0]
        label = line_split[1]
        feat = h5[key + '.npy'][:]
        if label == '1':
            anno_npy_path = os.path.join(mask_dir, key + '.npy')
            anno = np.load(anno_npy_path)
            labels.append('Abnormal')
        else:
            anno = np.zeros(int(line_split[-1]))
            labels.append('Normal')
        output_feats.append(feat)
        annos.append(anno)
        names.append(key)
    if return_names == True:
        return output_feats, labels, annos, names
    return output_feats,labels,annos

class SH_Train_Origin_Dataset_tenCrop(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, n_patch, sample, d_model, pseudo_labels_path=None):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch
        self.d_model = d_model
        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []
        lab_dict = {'Normal': 0, 'Abnormal': 1}
        h5 = h5py.File(self.h5_path, 'r')
        lines = open(self.train_txt, 'r').readlines()
        self.norm_keys = []
        self.abnorm_keys = []
        for line in lines:
            line_split = line.strip().split(',')
            label = int(line_split[-1])
            key = line_split[0]
            if label == 0:
                self.norm_feats.append(np.reshape(h5[key + '.npy'][:], (-1, 10, self.n_patch, self.d_model)))
                if self.pseudo_labels != None:
                    self.norm_keys.append(key + '.npy')
            else:
                self.abnorm_feats.append(np.reshape(h5[key + '.npy'][:], (-1, 10, self.n_patch, self.d_model)))
                if self.pseudo_labels != None:
                    self.abnorm_keys.append(key + '.npy')
    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat, labs, vid_type='Normal'):
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)

        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]

        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])
        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter][:-4]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter][:-4]]
        else:
            norm_labs = abnorm_labs = None
        crop_i = random.randint(0, 9)
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter][:, crop_i, :, :], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter][:, crop_i, :, :], abnorm_labs, vid_type='Abnormal')
        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:

            return torch.from_numpy(norm_feat).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat).float(), torch.from_numpy(abnorm_labs).float(), crop_i

class SH_Train_Origin_Dataset_MutualTraining(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, n_patch, sample, pseudo_labels_path=None):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch

        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []
        lab_dict = {'Normal': 0, 'Abnormal': 1}
        h5 = h5py.File(self.h5_path, 'r')
        lines = open(self.train_txt, 'r').readlines()
        self.norm_keys = []
        self.abnorm_keys = []
        for line in lines:
            line_split = line.strip().split(',')
            label = int(line_split[-1])
            key = line_split[0]
            if label == 0:
                self.norm_feats.append(key + '.npy')
                if self.pseudo_labels != None:
                    self.norm_keys.append(key + '.npy')
            else:
                self.abnorm_feats.append(key + '.npy')
                if self.pseudo_labels != None:
                    self.abnorm_keys.append(key + '.npy')


    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat_name, labs, vid_type='Normal'):
        with h5py.File(self.h5_path, 'r') as h5_file:
            feat = h5_file[feat_name][:]
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)

        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]

        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])

        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter]]
        else:
            norm_labs = abnorm_labs = None
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter], abnorm_labs, vid_type='Abnormal')

        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:
            return torch.from_numpy(norm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(abnorm_labs).float()

def shanghaitech_test_tenCrop(txt_path, mask_dir, h5_file, n_patch, d_model, return_names=False):
    lines = open(txt_path,'r').readlines()
    annos = []
    labels = []
    names = []
    h5 = h5py.File(h5_file, 'r')
    output_feats = []
    for line in lines:
        line_split = line.strip().split(',')
        key = line_split[0]
        label = line_split[1]
        feat = np.reshape(h5[key + '.npy'][:], (-1, 10, n_patch, d_model))
        if label == '1':
            anno_npy_path = os.path.join(mask_dir, key + '.npy')
            anno = np.load(anno_npy_path)
            labels.append('Abnormal')
        else:
            anno = np.zeros(int(line_split[-1]))
            labels.append('Normal')
        output_feats.append(feat)
        annos.append(anno)
        names.append(key)
    if return_names == True:
        return output_feats, labels, annos, names
    return output_feats,labels,annos

class UCF_Train_Origin_Dataset(Dataset):
    def __init__(self, part_num, part_len, frames_per_clip, h5_path, train_txt, n_patch, sample, pseudo_labels_path=None,
                 d_model=4096, crop_return=False):
        self.part_num = part_num
        self.part_len = part_len
        self.frames_per_clip = frames_per_clip
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch
        self.crop_return = crop_return
        self.d_model = d_model
        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_video_list(self.train_txt)
        self.shuffle_keys()

    def load_video_list(self, annotation_txt):
        self.norm_video_names_list = []
        self.abnorm_video_names_list = []
        self.video_number_frames = {}
        for line in open(annotation_txt, 'r').readlines():
            key = line.strip().split(" ")[0].split('/')[-1].split('.')[0]
            n_frames = int(line.strip().split(" ")[1])
            if key.split("_")[0] == "Normal":
                self.norm_video_names_list.append(key)
            else:
                self.abnorm_video_names_list.append(key)
            self.video_number_frames[key] = n_frames

    def __len__(self):
        return min(len(self.norm_video_names_list), len(self.abnorm_video_names_list))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_video_names_list))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_video_names_list))

    def sample_feat(self, video_name, labs, vid_type='Normal'):
        with h5py.File(self.h5_path, 'r') as h5_file:
            if self.crop_return == False:
                feat = h5_file[video_name+".npy"][:]
            else:
                feat = h5_file[video_name+".npy"][:].reshape((-1, 10, self.n_patch, self.d_model))
                crop_i = random.randint(0, 9)
                feat = feat[:, crop_i, :, :]

        if feat.shape[0] <= self.part_len:
            feat = np.repeat(feat, 2, axis=0)
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)
        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]
        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1, dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])
        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_video_names_list[norm_iter]+'.npy']
            abnorm_labs = self.pseudo_labels[self.abnorm_video_names_list[abnorm_iter]+'.npy']
        else:
            norm_labs = abnorm_labs = None
        norm_feat, norm_labs = self.sample_feat(self.norm_video_names_list[norm_iter], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_video_names_list[abnorm_iter], abnorm_labs, vid_type='Abnormal')
        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:
            return torch.from_numpy(norm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(abnorm_labs).float()

def UCF_train(line, data_h5_file_path, frames_per_clip=16, return_name=False):
    with h5py.File(data_h5_file_path, 'r') as data_h5_file:
        output_feats = []
        line_split = line.strip().split(' ')
        key = line_split[0].split('/')[1].split('.')[0]
        n_frames = int(line_split[1])
        output_feats = data_h5_file[key + ".npy"][:]

    if return_name ==True:
        return output_feats, n_frames, key
    return output_feats, n_frames

def UCF_test(line, data_h5_file_path, gt_h5_file_path, frames_per_clip=16, return_name=False):
    with h5py.File(data_h5_file_path, 'r') as data_h5_file:
        with h5py.File(gt_h5_file_path, 'r') as gt_h5_file:
            output_feats = []
            line_split = line.strip().split(' ')
            key = line_split[0].split('/')[1].split('.')[0]
            n_frames = int(line_split[1])
            label = line_split[2]
            if label == 'Normal':
                anno = np.zeros(int(line_split[1]))
            else:
                anno = gt_h5_file[key + '.npy'][:]
            output_feats = data_h5_file[key + ".npy"][:]
    if return_name ==True:
        return output_feats, anno, n_frames, key
    return output_feats, anno, n_frames

def UCF_test_tenCrop(line, data_h5_file_path, gt_h5_file_path, frames_per_clip=16, return_name=False):
    with h5py.File(data_h5_file_path, 'r') as data_h5_file:
        with h5py.File(gt_h5_file_path, 'r') as gt_h5_file:
            output_feats = []
            line_split = line.strip().split(' ')
            key = line_split[0].split('/')[1].split('.')[0]
            n_frames = int(line_split[1])
            label = line_split[2]
            if label == 'Normal':
                anno = np.zeros(int(line_split[1]))
            else:
                anno = gt_h5_file[key + '.npy'][:]
            output_feats = data_h5_file[key + ".npy"][:]
    if return_name == True:
        return output_feats, anno, n_frames, key
    return output_feats, anno, n_frames


class UBnormal_Train_Origin_Dataset(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, n_patch, sample, pseudo_labels_path=None):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch

        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []
        with h5py.File(self.h5_path, "r") as h5:
            lines = open(self.train_txt, 'r').readlines()
            self.norm_keys = []
            self.abnorm_keys = []
            for line in lines:
                key = line.strip().split(",")[0]
                if key.split("_")[0] == "normal":
                    self.norm_feats.append(h5[key + '.npy'][:])
                    if self.pseudo_labels != None:
                        self.norm_keys.append(key + '.npy')
                else:
                    self.abnorm_feats.append(h5[key + '.npy'][:])
                    if self.pseudo_labels != None:
                        self.abnorm_keys.append(key + '.npy')
    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat, labs, vid_type='Normal'):
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)

        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]
        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])
        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter]]
        else:
            norm_labs = abnorm_labs = None
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter], abnorm_labs, vid_type='Abnormal')
        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:
            return torch.from_numpy(norm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :self.n_patch, :]).float(), torch.from_numpy(abnorm_labs).float()

def UBnormal_test(txt_path, mask_dir, h5_file, return_names=False):
    with h5py.File(h5_file, 'r') as h5:
        lines = open(txt_path, 'r').readlines()
        annos = []
        labels = []
        names = []
        output_feats = []
        for line in lines:
            key = line.strip().split(",")[0]
            n_frames = line.strip().split(",")[1]
            feat = h5[key + '.npy'][:]
            if key.split("_")[0] == "abnormal":
                anno_npy_path = os.path.join(mask_dir, key + '.npy')
                anno = np.load(anno_npy_path)
                labels.append('Abnormal')
            else:
                anno = np.zeros(int(n_frames))
                labels.append('Normal')
            output_feats.append(feat)
            annos.append(anno)
            names.append(key)
        if return_names == True:
            return output_feats, labels, annos, names
        return output_feats, labels, annos

class UBnormal_Train_Origin_Dataset_tenCrop(Dataset):
    def __init__(self, part_num, part_len, h5_path, train_txt, n_patch, sample, d_model, pseudo_labels_path=None):
        self.part_num = part_num
        self.part_len = part_len
        self.train_txt = train_txt
        self.h5_path = h5_path
        self.sample = sample
        self.n_patch = n_patch
        self.d_model = d_model

        if pseudo_labels_path != None:
            if os.path.exists(pseudo_labels_path):
                print ("Pseudo labels load successful.")
                self.pseudo_labels = np.load(pseudo_labels_path, allow_pickle=True).tolist()
            else:
                print ("Can NOT open the pseudo labels file!")
                exit(-1)
        else:
            self.pseudo_labels = None
        self.load_feat()
        self.shuffle_keys()

    def load_feat(self):
        self.norm_feats = []
        self.abnorm_feats = []

        with h5py.File(self.h5_path, "r") as h5:
            lines = open(self.train_txt, 'r').readlines()
            self.norm_keys = []
            self.abnorm_keys = []
            for line in lines:
                key = line.strip().split(",")[0]
                if key.split("_")[0] == "normal":
                    self.norm_feats.append(np.reshape(h5[key + '.npy'][:], (-1, 10, self.n_patch, self.d_model)))
                    if self.pseudo_labels != None:
                        self.norm_keys.append(key + '.npy')
                else:
                    self.abnorm_feats.append(np.reshape(h5[key + '.npy'][:], (-1, 10, self.n_patch, self.d_model)))
                    if self.pseudo_labels != None:
                        self.abnorm_keys.append(key + '.npy')
    def __len__(self):
        return min(len(self.norm_feats), len(self.abnorm_feats))

    def shuffle_keys(self):
        self.norm_iters = np.random.permutation(len(self.norm_feats))
        self.abnorm_iters = np.random.permutation(len(self.abnorm_feats))

    def sample_feat(self, feat, labs, vid_type='Normal'):
        feat_len = feat.shape[0]
        if type(labs) == type(None):
            if vid_type == 'Normal':
                labs = np.zeros([feat_len, 1], dtype=np.float32)
            else:
                labs = np.ones([feat_len, 1], dtype=np.float32)

        else:
            if labs.shape.__len__() == 2 and labs.shape[-1] == 2:
                labs = labs[:, -1]

        if self.sample == 'uniform':
            if (feat_len - self.part_len) // (self.part_num + 1) < 1:
                move = 0
            else:
                move = np.random.randint((feat_len - self.part_len) // (self.part_num + 1))
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int) + move
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
        else:
            chosen = np.linspace(0, feat_len - self.part_len, num=self.part_num + 1, dtype=int)
            chosen = chosen.repeat(self.part_len).reshape([-1, self.part_len]) + np.arange(0, self.part_len, 1,
                                                                                           dtype=int)
            if chosen[1, 0] - chosen[0, 0] == 0:
                move = 0
            else:
                move = np.random.randint(0, chosen[1, 0] - chosen[0, 0], [self.part_num + 1]).repeat(
                    self.part_len).reshape([-1, self.part_len])
            chosen = chosen + move
        chosen = chosen.reshape([-1])
        return feat[chosen[:self.part_num * self.part_len], :], labs[chosen[:self.part_num * self.part_len]]

    def __getitem__(self, item):
        norm_iter = self.norm_iters[item]
        abnorm_iter = self.abnorm_iters[item]
        if self.pseudo_labels != None:
            norm_labs = self.pseudo_labels[self.norm_keys[norm_iter]]
            abnorm_labs = self.pseudo_labels[self.abnorm_keys[abnorm_iter]]
        else:
            norm_labs = abnorm_labs = None

        crop_i = random.randint(0, 9)
        norm_feat, norm_labs = self.sample_feat(self.norm_feats[norm_iter][:, crop_i, :, :], norm_labs, vid_type="Normal")
        abnorm_feat, abnorm_labs = self.sample_feat(self.abnorm_feats[abnorm_iter][:, crop_i, :, :], abnorm_labs, vid_type='Abnormal')
        if self.n_patch == 1:
            return torch.from_numpy(norm_feat[:, :]).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat[:, :]).float(), torch.from_numpy(abnorm_labs).float()
        else:

            return torch.from_numpy(norm_feat).float(), torch.from_numpy(norm_labs).float(), \
                   torch.from_numpy(abnorm_feat).float(), torch.from_numpy(abnorm_labs).float(), crop_i

def UBnormal_test_tenCrop(txt_path, mask_dir, h5_file, n_patch, d_model, return_names=False):
    with h5py.File(h5_file, 'r') as h5:
        lines = open(txt_path, 'r').readlines()
        annos = []
        labels = []
        names = []
        output_feats = []
        for line in lines:
            key = line.strip().split(",")[0]
            n_frames = line.strip().split(",")[1]
            feat = np.reshape(h5[key + '.npy'][:], (-1, 10, n_patch, d_model))
            if key.split("_")[0] == "abnormal":
                anno_npy_path = os.path.join(mask_dir, key + '.npy')
                anno = np.load(anno_npy_path)
                labels.append('Abnormal')
            else:
                anno = np.zeros(int(n_frames))
                labels.append('Normal')
            output_feats.append(feat)
            annos.append(anno)
            names.append(key)

    if return_names == True:
        return output_feats, labels, annos, names
    return output_feats,labels,annos
