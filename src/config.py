
import torch

from src.utils import Print


class DataConfig():
    def __init__(self, file=None, idx="data_config"):
        """ data configurations """
        self.idx = idx
        self.with_esa = True
        self.path = {}

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if "with_esa" in key:             self.with_esa = value
                elif "path" in key:               self.path[key.split("_")[0]] = value
                else: sys.exit("# ERROR: invalid key [%s] in data-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["with_esa", self.with_esa])
        configs.append(["path", self.path])
        return configs


import os
import sys
import json

class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.ds_out_channels = None
        self.block_kernel_size = None
        self.lstm_hidden_size = None
        self.lstm_num_layers = None
        self.kmer = None
        self.cuda = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file):
                sys.exit("model-config [%s] does not exists" % file)
            else:
                cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "ds_out_channels":          self.ds_out_channels = value
                elif key == "block_kernel_size":     self.block_kernel_size = value
                elif key == "lstm_hidden_size":      self.lstm_hidden_size = value
                elif key == "lstm_num_layers":       self.lstm_num_layers = value
                elif key == "kmer":                  self.kmer = value
                elif key == "cuda":                  self.cuda = value
                else:
                    sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def get_config(self):
        configs = []
        if self.ds_out_channels is not None: configs.append(["ds_out_channels", self.ds_out_channels])
        if self.block_kernel_size is not None: configs.append(["block_kernel_size", self.block_kernel_size])
        if self.lstm_hidden_size is not None: configs.append(["lstm_hidden_size", self.lstm_hidden_size])
        if self.lstm_num_layers is not None: configs.append(["lstm_num_layers", self.lstm_num_layers])
        if self.kmer is not None: configs.append(["kmer", self.kmer])
        if self.cuda is not None: configs.append(["cuda", self.cuda])

        return configs



class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False):
        """ run configurations """
        self.idx = idx
        self.eval = eval
        self.batch_size = None
        self.num_epochs = None
        self.learning_rate = None
        self.weight_decay = None
        self.dropout_rate = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("run-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "batch_size":                   self.batch_size = value
                elif key == "num_epochs":                   self.num_epochs = value
                elif key == "learning_rate":                self.learning_rate = value
                elif key == "weight_decay":                 self.weight_decay = value
                elif key == "dropout_rate":                 self.dropout_rate = value
                else: sys.exit("# ERROR: invalid key [%s] in run-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["batch_size_eval", self.batch_size])
        if not self.eval:
            configs.append(["num_epochs", self.num_epochs])
            configs.append(["learning_rate", self.learning_rate])
            configs.append(["weight_decay", self.weight_decay])
            configs.append(["dropout_rate", self.dropout_rate])

        return configs


def print_configs(args, cfgs, device, output):
    Print(" ".join(['##### arguments #####']), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    if args["checkpoint"] is not None:
        Print(" ".join(['checkpoint: %s' % (args["checkpoint"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)