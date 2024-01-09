
""" Utility functions """

import os
import sys
import random
import datetime
import numpy as np

import torch
# from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else:
        time = None
        line = string

    print(line, file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_args(args):
    """ sanity check for arguments """
    if args["checkpoint"] is not None and not os.path.exists(args["checkpoint"]):
            sys.exit("checkpoint [%s] does not exists" % (args["checkpoint"]))


def set_output(args, string):
    """ set output configurations """
    output, save_prefix = sys.stdout, None

    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(args["output_path"] + "/" + string + ".txt", "a")

    return output, save_prefix

def indicators(set_labels, outputs):
    '''set output indicators'''
    set_num = len(set_labels)
    y_true = set_labels
    # print(y_true)
    outputs = outputs.flatten()
    y_predict = np.round(outputs)
    # print(y_predict)
    TP, TN, FP, FN = 0, 0, 0, 0
    epsilon = 1e-7  # 为了防止分母为0会报错
    for i in range(set_num):
        if y_predict[i] == 1 and y_true[i] == 1:
            TP += 1
        if y_predict[i] == 0 and y_true[i] == 0:
            TN += 1
        if y_predict[i] == 1 and y_true[i] == 0:
            FP += 1
        if y_predict[i] == 0 and y_true[i] == 1:
            FN += 1
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    acc = (TP + TN) / (TP + TN + FP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    neg_precision = TN /(TN + FN + epsilon)
    auc = roc_auc_score(y_true, y_predict)
    return f1, acc, precision, recall, specificity, neg_precision, auc

def get_parameter_number(model):
    #  打印模型参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)
    # return total_num, trainable_num

def get_data_from_configs(data_cfg):

    data, labels = [], []
    FILE = open(data_cfg.path['val'], "r")
    # FILE = open('/home/zhangjialin/TargetNet-master_acc/data/text0.txt', "r")
    lines = FILE.readlines()
    for l, line in enumerate(lines[1:]):
        # data.append(line)
        tokens = line.strip().split("\t")
        data.append(line)
        label = float(tokens[4]) if len(tokens) > 4 else 0
        labels.append(label)

    FILE.close()

    return data, labels

def get_data(data_cfg, split_idx=None):

    data, labels = [], []
    FILE = open(data_cfg.path[split_idx], "r")
    lines = FILE.readlines()
    for l, line in enumerate(lines[1:]):
        tokens = line.strip().split("\t")
        data.append(line)
        label = float(tokens[4]) if len(tokens) > 4 else 0
        labels.append(label)

    FILE.close()

    return data, labels


