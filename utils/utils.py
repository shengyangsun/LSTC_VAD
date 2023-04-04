import numpy as np
import logging
import os
import random
import os
import torch
import time
import cv2
import torch.nn.functional as F


def decode_imgs(frames, patch_per_height, patch_per_width):
    new_frames = []
    for i, frame in enumerate(frames):
        img_patch_list = []
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        height, width = img.shape[0] // patch_per_height, img.shape[1] // patch_per_width
        for j in range(patch_per_height):
            for k in range(patch_per_width):
                img_patch = img[j*height:(j+1)*height, k*width:(k+1)*width, :]
                img_patch_list.append(img_patch)
        new_frames.append(img_patch_list)
    return new_frames

def get_video_names(txt_path, abnormal=True, normal=True):
    video_name_list = []
    lines = open(txt_path, 'r').readlines()
    for line in lines:
        line_split = line.strip().split(',')
        label = int(line_split[1])
        key = line_split[0]
        if label == 1:
            if abnormal == True:
                video_name_list.append(key)
        else:
            if normal == True:
                video_name_list.append(key)
    return video_name_list

def get_video_names_UBnormal(txt_path, abnormal=True, normal=True):
    video_name_list = []
    lines = open(txt_path, 'r').readlines()
    for line in lines:
        line_split = line.strip().split(',')
        key = line_split[0]
        label = key.split("_")[0]
        if label == "abnormal":
            if abnormal == True:
                video_name_list.append(key)
        else:
            if normal == True:
                video_name_list.append(key)
    return video_name_list

def get_video_names_frames_labels_UCF(txt_path):
    key_list = []
    n_frames_list = []
    for line in open(txt_path, 'r').readlines():
        key = line.strip().split(" ")[0].split('/')[-1].split('.')[0]
        n_frames = int(line.strip().split(" ")[1])
        key_list.append(key)
        n_frames_list.append(n_frames)
    return key_list, n_frames_list

def show_params(args):
    params=vars(args)
    keys=sorted(params.keys())

    for k in keys:
        print(k,'\t',params[k])

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def log_param(logger, args):
    params=vars(args)
    keys=sorted(params.keys())
    for k in keys:
        log_info = '{}\t{}'.format(k,params[k])
        print (log_info)
        logger.info(log_info)


def get_timestamp():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

def mkdir(dir):
    if not os.path.exists(dir):
        try:os.mkdir(dir)
        except:pass

def set_seeds(seed):
    print('set seed {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def random_perturb(v_len, num_segments):
    random_p = np.arange(num_segments) * v_len / num_segments
    for i in range(num_segments):
        if i < num_segments - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def weights_normal_init(model, dev=0.01):
    import torch
    from torch import nn

    # torch.manual_seed(2020)
    # torch.cuda.manual_seed_all(2020)
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias!=None:
                    m.bias.data.fill_(0)

def log_setting(args):
    get_timestamp()
    if args.dataset == "SHT":
        logger_dir = "/data/ssy/code/VAD_ST/log/SHT"
    elif args.dataset == "UCF":
        logger_dir = "/data/ssy/code/VAD_ST/log/UCF"
    elif args.dataset == "UBnormal":
        logger_dir = "/data/ssy/code/VAD_ST/log/UBnormal"
    else:
        logger_dir = "/data/ssy/code/VAD_ST/log/other"
    mkdir(logger_dir)
    if args.model == "Temporal":
        param_str = '{}_{}_seed_{}_bs_{}_encoderLR_{}_classifierLR_{}_nPatch_{}_nHead_{}_nLayer_{}_{}'.\
            format(args.data_crop, args.type, args.seed, args.batch_size, args.lr_encoder, args.lr_classifier, args.n_patch, args.n_head, args.n_layers, get_timestamp())
    elif args.model == "Spatio":
        param_str = '{}_{}_seed_{}_bs_{}_encoderLR_{}_regressorLR_{}_nPatch_{}_nHead_{}_nLayer_{}_{}'. \
            format(args.data_crop, args.type, args.seed, args.batch_size, args.lr_encoder, args.lr_regressor, args.n_patch, args.n_head, args.n_layers, get_timestamp())
    logger_path = os.path.join(logger_dir, '{}.log'.format(param_str))
    logger = get_logger(logger_path)
    logger.info('This model starts at time {}'.format(get_timestamp()))
    log_param(logger, args)
    return logger

