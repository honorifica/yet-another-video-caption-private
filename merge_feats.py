import numpy as np
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
from tqdm import tqdm
import threading

if __name__ == '__main__':
    if not os.path.exists("data/feats"):
        os.mkdir("data/feats")

    if not os.path.exists("data/feats/hybrid"):
        os.mkdir("data/feats/hybrid")

    video_list = glob.glob(os.path.join('data/video', '*.avi'))
    a_list = [i.replace('video','feats/resnet152').replace('avi','npy') for i in video_list]
    b_list = [i.replace('video','feats/c3d').replace('avi','npy') for i in video_list]
    c_list = [i.replace('video','feats/hybrid').replace('avi','npy') for i in video_list]
    
    for (a,b,c) in tqdm(zip(a_list, b_list, c_list)):
        an = np.load(a)
        bn = np.load(b)
        cn = np.concatenate((an,bn), axis=-1)
        np.save(c, cn)