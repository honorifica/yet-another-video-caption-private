import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from c3d.opts import parse_opts
from c3d.model import generate_model
from c3d.mean import get_mean
from c3d.classify import classify_video

from tqdm import tqdm

import threading

def __fuck(src_dest_list, tid=0):
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 25
    opt.n_classes = 400

    model = generate_model(opt)
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('c3d/class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'

    subprocess.call('mkdir result', shell=True)
    
    for src_dest in tqdm(src_dest_list):
        src = src_dest[0]
        dest =  src_dest[1]
        if os.path.exists('c3d/tmp%d'%tid):
            subprocess.call('rm -rf c3d/tmp%d'%tid, shell=True)
        subprocess.call('mkdir c3d/tmp%d'%tid, shell=True)
        subprocess.call('ffmpeg -i {} c3d/tmp{}/image_%05d.jpg -loglevel quiet'.format(src,tid),
                                shell=True)
        result = classify_video('c3d/tmp%d'%tid, '', class_names, model, opt)
        rnp = [i["features"] for i in result["clips"]]
        subprocess.call('rm -rf c3d/tmp%d'%tid, shell=True)
        rnp = np.array(rnp)
        np.save(dest, rnp)

    if os.path.exists('tmp'):
        subprocess.call('rm -rf c3d/tmp%d'%tid, shell=True)
    

def fuck(src_dest_list):
#     NP = 4
#     th_pool = []
#     for i in range(NP):
#         th = threading.Thread(target=__fuck, args=(src_dest_list,i,))
#         th_pool.append(th)
#     for i in th_pool:
#         i.start()
#     for i in th_pool:
#         i.join()
    __fuck(src_dest_list)
        
if __name__=="__main__":
    fuck([['videos/video0.mp4', '0.npy']])