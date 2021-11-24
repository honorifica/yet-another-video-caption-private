# 这个程序用于从 dataset 中抽取需要的数据到 data 中，并自动生成符合格式要求的 data/input.json
# 警告：执行该脚本会导致 data 下的数据丢失！！！

import os
import json
import shutil
from tqdm import tqdm


def cleanDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


NTRAIN = 3200
NVAL = 200
NTEST = 1000
cleanDir('data')
print("步骤1：扫描输入并生成 input.json")
fp = open("dataset/train/info.json", "r")
train_info = json.load(fp)
train_videos = train_info["videos"][:NTRAIN+NVAL]
train_captions = train_info["sentences"][:(NTRAIN+NVAL)*5]
for i in range(NVAL):
    train_videos[i+NTRAIN]["split"] = "val"
train_video_names = []
for i in train_videos:
    train_video_names.append(i["video_id"])
test_video_filenames = os.listdir("dataset/test/video")
test_video_names = [i.split(".")[0] for i in test_video_filenames]
test_video_names = test_video_names[:NTEST]
for i in test_video_names:
    train_videos.append({'category': 0, 'url': '', 'video_id': i, 'start time': '',
                        'end time': '', 'split': 'test', 'id': int(i.split('_')[1])})
export_json = {"videos": train_videos, "sentences": train_captions}
fp = open("data/input.json", "w")
json.dump(export_json, fp)
fp.close()
print("已完成 :)")
print("步骤2：拷贝视频文件（目前仅拷贝 video，忽略多模态信息）")
vl1 = test_video_filenames[:NTEST]
vl2 = [i + ".avi" for i in train_video_names]
cleanDir("data/video")
for i in tqdm(vl1):
    shutil.copyfile("dataset/test/video/"+i, "data/video/"+i)
for i in tqdm(vl2):
    shutil.copyfile("dataset/train/video/"+i, "data/video/"+i)
print("已完成 :)")
print("Bye~")
