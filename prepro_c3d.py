import shutil
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

import threading
import c3d.fuck


if __name__ == '__main__':
    if not os.path.exists("data/feats"):
        os.mkdir("data/feats")

    if not os.path.exists("data/feats/c3d"):
        os.mkdir("data/feats/c3d")

    video_list = glob.glob(os.path.join('data/video', '*.avi'))
    output_list = [i.replace('video','feats/c3d').replace('avi','npy') for i in video_list]
    
    src_dest_list = [list(i) for i in list(zip(video_list, output_list))]
    
    c3d.fuck.fuck(src_dest_list)