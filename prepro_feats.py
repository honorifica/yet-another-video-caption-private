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

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def exone(params, model, load_image_fn, dir_fc, video):
    video_id = video.split("/")[-1].split(".")[0].split("\\")[-1]
    dst = params['model'] + '/' + video_id
    extract_frames(video, dst)

    image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
    samples = np.round(np.linspace(
        0, len(image_list) - 1, params['n_frame_steps']))
    image_list = [image_list[int(sample)] for sample in samples]
    images = torch.zeros((len(image_list), C, H, W))
    for iImg in range(len(image_list)):
        img = load_image_fn(image_list[iImg])
        images[iImg] = img
    with torch.no_grad():
        fc_feats = model(images.cuda()).squeeze()
    img_feats = fc_feats.cpu().numpy()
    # Save the inception features
    outfile = os.path.join(dir_fc, video_id + '.npy')
    np.save(outfile, img_feats)
    # cleanup
    shutil.rmtree(dst)

def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    video_list = glob.glob(os.path.join(params['video_path'], '*.avi'))

    # NP 路并行！！！
    NP = 1
    vll = []
    while len(video_list)>0:
        vll.append(video_list[:NP])
        video_list = video_list[NP:]

    print("已启用线程级并行。若出现显存不足，请调小代码中参数 NP")

    for vl in tqdm(vll):
        thread_pool = []
        for i in vl:
            thread_pool.append(threading.Thread(target=exone,args=(params,model,load_image_fn,dir_fc,i)))
        for i in thread_pool:
            i.start()
        for i in thread_pool:
            i.join()

    print("特征抽抽抽已完成 :)")


if __name__ == '__main__':
    if not os.path.exists("data/feats"):
        os.mkdir("data/feats")

    if not os.path.exists("data/feats/resnet152"):
        os.mkdir("data/feats/resnet152")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats/resnet152', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=40,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='data/video', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'inception_v3':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv3(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'resnet152':
        C, H, W = 3, 224, 224
        model = pretrainedmodels.resnet152(pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    elif params['model'] == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(
            num_classes=1000, pretrained='imagenet')
        load_image_fn = utils.LoadTransformImage(model)

    else:
        print("doesn't support %s" % (params['model']))

    model.last_linear = utils.Identity()
    model = nn.DataParallel(model)

    model = model.cuda()
    extract_feats(params, model, load_image_fn)
