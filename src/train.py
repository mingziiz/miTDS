import sys
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

import transformers
from src.utils import Print, indicators
from transformers import logging
# logging.set_verbosity_warning()

class Trainer():
    """ train / eval helper class """
    def __init__(self, model):
        self.model = model
        self.optim = None
        self.scheduler = None

        # initialize logging parameters
        self.epoch = 0.0
        self.best_loss = None
        self.logger_train = Logger()
        self.logger_eval  = Logger()


    def train(self, batch, device, model_cfg):
        # training of the model
        batch = set_device(batch, device)

        self.model.train()
        self.optim.zero_grad()
        inputs, labels, set_idxs, input_ids, attention_masks, token_type_ids = batch
        outputs = self.model(inputs, input_ids, attention_masks, token_type_ids)
        loss = get_loss(outputs, labels)
        loss.backward()
        self.optim.step()

        outputs = torch.sigmoid(outputs)
        self.logger_train.update(len(outputs), loss.item())
        self.logger_train.keep(outputs, set_idxs)

    def evaluate(self, batch, device, model_cfg):
        # evaluation of the model
        batch = set_device(batch, device)
        self.model.eval()
        with torch.no_grad():
            inputs, labels, set_idxs, input_ids, attention_masks, token_type_ids = batch
            outputs = self.model(inputs, input_ids, attention_masks, token_type_ids)
            loss = get_loss(outputs, labels)

        # logging
        outputs = torch.sigmoid(outputs)
        self.logger_eval.update(len(outputs), loss.item())
        self.logger_eval.keep(outputs, set_idxs)


    def scheduler_step(self):
        # scheduler_step
        self.scheduler.step(self.logger_eval.get_loss())

    def save_model(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return

        loss = self.logger_eval.get_loss()

        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.model.state_dict(), save_prefix + "/TargetNet.pt")
            torch.save(self.model.bert.state_dict(), save_prefix + "/bert_finetune.bin")


    def load_model(self, checkpoint, output):
        # load state_dicts from checkpoint """
        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict)

    def save_outputs(self, idx, save_prefix):
        # save validation output
        OUT = open(save_prefix + "/%s_outputs.txt" % (idx), "w")
        OUT.write("\t".join(["set_idx", "output"]) + "\n")
        for i in range(len(self.logger_eval.outputs)):
            OUT.write("\t".join([str(i), "%f" % self.logger_eval.outputs[i]]) + "\n")

            if i % 5 == 0:
                print('# {} {:.1%}'.format(idx, i / len(self.logger_eval.outputs)), end='\r', file=sys.stderr)
        OUT.write('F1 Score: %f' % self.logger_eval.f1_score + '\n')
        OUT.write('Accuracy: %f' % self.logger_eval.accuracy + '\n')
        OUT.write('Precision: %f' % self.logger_eval.precision + '\n')
        OUT.write('Recall: %f' % self.logger_eval.recall+ '\n')
        OUT.write('Specificity: %f' % self.logger_eval.specificity + '\n')
        OUT.write('Neg_precision: %f' % self.logger_eval.neg_precision + '\n')
        print(' ' * 150, end='\r', file=sys.stderr)

        OUT.close()
        self.log_reset()

    def set_device(self, device):
        # set gpu configurations
        self.model = self.model.to(device)

    def set_optim_scheduler(self, run_cfg, params, model, iterator_train):
        # set optim and scheduler for training
        aa = params
        optim, scheduler = get_optim_scheduler(run_cfg, params, model, iterator_train)
        self.optim = optim
        self.scheduler = scheduler

    def aggregate(self, set_num):
        # aggregate kept outputs, labels, set_idxs
        self.logger_eval.aggregate(set_num)

    def get_headline(self):
        # get a headline for logging
        headline = ["ep", "split", "loss","acc", "|", "loss", "acc"]
        return "\t".join(headline)

    def log(self, idx, output,set_train_labels,set_val_labels):
        # logging

        log = ["%03d" % self.epoch, "train",
               "%.4f" % self.logger_train.get_loss(),"%.4f" % self.logger_train.aggregate(set_train_labels),
               "|", idx, "%.4f" % self.logger_eval.get_loss(), "%.4f" % self.logger_eval.aggregate(set_val_labels)
               ]
        Print("\t".join(log), output)
        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()

    def get_val_loss(self):
        return self.logger_eval.get_loss()


class Logger():
    """ Logger class """
    def __init__(self):
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.set_idxs = []
        self.log = []

    def update(self, total, loss):
        # update logger for current mini-batch
        self.total += total
        self.loss += loss * total

    def keep(self, outputs, set_idxs):
        # keep outputs, labels, and set_idxs for future computations
        self.outputs.append(outputs.cpu().detach().numpy())
        self.set_idxs.append(set_idxs.cpu().detach().numpy())

    def get_loss(self):
        # get current averaged loss
        loss = self.loss / self.total
        return loss

    def aggregate(self, set_labels):
        # aggregate kept labels and outputs
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.set_idxs = np.concatenate(self.set_idxs, axis=0)

        set_num = len(set_labels)
        # if len(self.set_idxs) != set_num:
        set_outputs = np.zeros(set_num, np.float32)
        for i in range(set_num):
            idxs = self.set_idxs == i
            if np.max(idxs) > 0: set_outputs[i] = np.max(self.outputs[idxs])
        self.outputs = set_outputs
        self.set_idxs = np.zeros(set_num, np.float32)
        self.f1_score, self.accuracy, self.precision, self.recall, self.specificity, self.neg_precision, self.auc = indicators(set_labels,self.outputs)
        return self.accuracy

    def reset(self):
        # reset logger
        self.total = 0.0
        self.loss = 0.0
        self.outputs = []
        self.set_idxs = []
        self.log = []


def get_optim_scheduler(cfg, params,model, iterator_train):
    """ configure optim and scheduler """
    optim = torch.optim.Adam([
            {'params': params[0], 'weight_decay': cfg.weight_decay},
            {'params': params[1], 'weight_decay': 0, 'lr':cfg.learning_rate},
            {'params': params[2], 'weight_decay': 1e-4, 'lr':0.001}],
            lr=cfg.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", 0.2, 5)

    return optim, scheduler


def get_loss(outputs, labels):
    """ get (binary) cross entropy loss """
    loss = -torch.mean(labels * F.logsigmoid(outputs) + (1 - labels) * F.logsigmoid(-outputs))

    return loss


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
