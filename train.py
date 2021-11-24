import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
import os
import argparse
from pandas.core.frame import DataFrame
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize

torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()
torch.cuda.empty_cache()


def convertToCocoFormat(data_frame):
    answers = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in answers:
            answers[row[1]].append(
                {'image_id': row[1], 'cap_id': len(answers[row[1]]), 'caption': row[0]})
        else:
            answers[row[1]] = []
            answers[row[1]].append(
                {'image_id': row[1], 'cap_id': len(answers[row[1]]), 'caption': row[0]})
    return answers


def validate(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()
    answer_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])

    answers = convertToCocoFormat(answer_dataframe)
    scores = []
    seq_outputs = {}
    for data in loader:
        input_features = data['fc_feats'].cuda()
        video_ids = data['video_ids']
        with torch.no_grad():
            seq_probs, seq_preds = model(
                input_features, mode='inference', opt=opt)

        seq_decodeds = utils.decode_sequence(vocab, seq_preds)

        for idx, seq in enumerate(seq_decodeds):
            video_id = video_ids[idx]
            seq_outputs[video_id] = [{'image_id': video_id, 'caption': seq}]

    valid_score = scorer.score(answers, seq_outputs, seq_outputs.keys())
    scores.append(valid_score)

    score_bleu4 = valid_score["Bleu_4"]
    score_meteor = valid_score["METEOR"]
    score_rougel = valid_score["ROUGE_L"]
    score_cider = valid_score["CIDEr"]
    score_sum = score_bleu4 + score_meteor + score_rougel + score_cider
    print("  验证集 coco 得分：", "%.6f = %.6f + %.6f + %.6f + %.6f" %
          (score_sum, score_bleu4, score_meteor, score_rougel, score_cider))

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(scores[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": seq_outputs, "scores": valid_score},
                  prediction_results)
    return score_sum


def train(dataloader_train, dataloader_val, dataset_val, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    #model = nn.DataParallel(model)
    ssm = 0
    smmid = 0
    for epoch in range(opt["epochs"]):
        model.train()
        lr_scheduler.step()

        iteration = 0

        total_loss_train = 0
        for data in dataloader_train:
            torch.cuda.synchronize()
            input_features = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            seq_probs, _ = model(input_features, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1
            total_loss_train += train_loss

        if epoch % 1 == 0:
            model_path = os.path.join(opt["checkpoint_path"],
                                      'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'model_score.txt')
            torch.save(model.state_dict(), model_path)
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, total_loss_train))

        total_loss_val = 0
        for data in dataloader_val:
            torch.cuda.synchronize()
            input_features = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            optimizer.zero_grad()
            seq_probs, _ = model(input_features, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])

            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            total_loss_val += train_loss

        print("  轮次", epoch, " train_loss =", total_loss_train /
              len(dataloader_train), " val_loss =", total_loss_val / len(dataloader_val))
        if epoch > 40:
            ss = validate(model, utils.LanguageModelCriterion(),
                          dataset_val, dataset_val.get_vocab(), opt)
            ssm = max(ssm, ss)
            if ssm == ss:
                smmid = epoch
            print(
                "HISTORY MAX                              **************************** ", ssm, smmid)


def main(opt):
    dataset_train = VideoDataset(opt, 'train')
    dataloader_train = DataLoader(
        dataset_train, batch_size=opt["batch_size"], shuffle=True)
    dataset_val = VideoDataset(opt, 'val')
    dataloader_val = DataLoader(
        dataset_val, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset_train.get_vocab_size()
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=opt["bidirectional"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder)
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    crit_rl = utils.RewardCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataloader_train, dataloader_val, dataset_val,  model, crit,
          optimizer, exp_lr_scheduler, opt, crit_rl)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
