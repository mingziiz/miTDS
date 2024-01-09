
import sys
import numpy as np
from Bio import pairwise2
import os
import pickle

import torch
from transformers import BertTokenizer, logging


class miRNA_CTS_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for miRNA-CTS pair data """
    def __init__(self, X, labels, set_idxs, set_labels, input_ids, attention_masks, token_type_ids):
        self.X = X
        self.labels = labels
        self.set_idxs = set_idxs
        self.set_labels = set_labels
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.X[i], self.labels[i], self.set_idxs[i], self.input_ids[i], self.attention_masks[i], self.token_type_ids[i]


def get_dataset_from_configs(data_cfg, model_cfg, split_idx=None):
    """ load miRNA-CTS dataset from config files """

    FILE = open(data_cfg.path[split_idx], "r")
    lines = FILE.readlines()
    FILE.close()

    X, labels, set_idxs, set_labels, input_ids, attention_masks, token_type_ids = [], [], [], [], [], [], []
    set_idx = 0

    for l, line in enumerate(lines[1:]):
        tokens = line.strip().split("\t")
        mirna_id, mirna_seq, mrna_id, mrna_seq = tokens[:4]
        label = float(tokens[4]) if len(tokens) > 4 else 0
        if split_idx in ["train", "val"] and tokens[5] != split_idx: continue

        mirna_seq = mirna_seq.upper().replace("T", "U")
        mrna_seq = mrna_seq.upper().replace("T", "U")
        mrna_rev_seq = reverse(mrna_seq)
        

        for pos in range(len(mrna_seq) - 40 + 1):
            mirna_esa, cts_rev_esa, esa_score = extended_seed_alignment(mirna_seq, mrna_rev_seq[pos:pos+40])
            if esa_score < 6: continue
            X.append(torch.from_numpy(encode_RNA(mirna_seq, mirna_esa,
                                                mrna_rev_seq[pos:pos+40], cts_rev_esa, data_cfg.with_esa)))
            labels.append(torch.from_numpy(np.array(label)).unsqueeze(0))
            set_idxs.append(torch.from_numpy(np.array(set_idx)).unsqueeze(0))
            mirna_dna = mirna_seq.upper().replace("U", "T")
            mrna_rev_dna = mrna_rev_seq.upper().replace("U", "T")
            seq_bert = (mirna_dna + (mrna_rev_dna[pos:pos + 40]))
            seq_kmer = seq2kmer(seq_bert, model_cfg.kmer)
            input_id, attention_mask, token_type_id = tokenizer(model_cfg, seq_kmer)
            input_ids.append(torch.from_numpy(np.array(input_id)))
            attention_masks.append(torch.from_numpy(np.array(attention_mask)))
            token_type_ids.append(torch.from_numpy(np.array(token_type_id)))


        set_labels.append(label)
        set_idx += 1


        if set_idx % 5 == 0:
            print('# {} {:.1%}'.format(split_idx, l / len(lines[1:])), end='\r', file=sys.stderr)
    print(' ' * 150, end='\r', file=sys.stderr)


    dataset = miRNA_CTS_dataset(X, labels, set_idxs, np.array(set_labels), input_ids, attention_masks, token_type_ids)

    return dataset


def encode_RNA(mirna_seq, mirna_esa, cts_rev_seq, cts_rev_esa, with_esa):
    """ one-hot encoder for RNA sequences with/without extended seed alignments """
    chars = {"A":0, "C":1, "G":2, "U":3}
    if not with_esa:
        x = np.zeros((len(chars) * 2, 40), dtype=np.float32)
        for i in range(len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i] = 1
        for i in range(len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
    else:
        chars["-"] = 4
        x = np.zeros((len(chars) * 2, 50), dtype=np.float32)
        for i in range(len(mirna_esa)):
            x[chars[mirna_esa[i]], 5 + i] = 1
        for i in range(10, len(mirna_seq)):
            x[chars[mirna_seq[i]], 5 + i - 10 + len(mirna_esa)] = 1
        for i in range(5):
            x[chars[cts_rev_seq[i]] + len(chars), i] = 1
        for i in range(len(cts_rev_esa)):
            x[chars[cts_rev_esa[i]] + len(chars), i + 5] = 1
        for i in range(15, len(cts_rev_seq)):
            x[chars[cts_rev_seq[i]] + len(chars), i + 5 - 15 + len(cts_rev_esa)] = 1

    return x


def reverse(seq):
    """ reverse the given sequence """
    seq_r = ""
    for i in range(len(seq)):
        seq_r += seq[len(seq) - 1 - i]
    return seq_r


score_matrix = {}  # Allow wobble
for c1 in 'ACGU':
    for c2 in 'ACGU':
        if (c1, c2) in [('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')]:
            score_matrix[(c1, c2)] = 1
        elif (c1, c2) in [('U', 'G'), ('G', 'U')]:
            score_matrix[(c1, c2)] = 1
        else:
            score_matrix[(c1, c2)] = 0


def extended_seed_alignment(mi_seq, cts_r_seq):
    """ extended seed alignment """
    alignment = pairwise2.align.globaldx(mi_seq[:10], cts_r_seq[5:15], score_matrix, one_alignment_only=True)[0]
    mi_esa = alignment[0]
    cts_r_esa = alignment[1]
    esa_score = alignment[2]
    return mi_esa, cts_r_esa, esa_score


def seq2kmer(seq, k):

    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = ' '.join(kmer)
    return kmers

def tokenizer(model_cfg, s):

    file_path = f'./pretrained_models/{model_cfg.kmer}-new-12w-0'

    tokenizer = BertTokenizer.from_pretrained(file_path, do_lower_case=False)
    seq = []
    seq.append(s)
    max_length = 68
    ids = tokenizer.batch_encode_plus(seq, padding=True, add_special_tokens=True)  # 有一个长度 如果达不到这个长度会补上，超出这个会截掉
    padding = [0] * (max_length - len(ids['input_ids'][0]))
    input_ids = ids['input_ids'][0] + padding
    token_type_ids = ids['token_type_ids'][0] + padding
    attention_masks = ids['attention_mask'][0] + padding

    return input_ids, attention_masks, token_type_ids

def process_pickle(split_idx, model_cfg, X, labels, set_idxs, set_labels, input_ids, attention_masks, token_type_ids):

    pickle_file = 'pickle/{}mer_{}.pickle'.format(model_cfg.kmer, split_idx)
    if not os.path.isfile(pickle_file):  # 判断是否存在此文件，若无则存储
        print('Saving data to {}mer_pickle file...'.format(split_idx))
        try:
            with open('pickle/{}mer_{}.pickle'.format(model_cfg, split_idx), 'wb') as pfile:
                pickle.dump(
                    {
                        'X': X,
                        'labels': labels,
                        'set_idxs': set_idxs,
                        'set_labels': set_labels,
                        'input_ids': input_ids,
                        'attention_masks': attention_masks,
                        'token_type_ids': token_type_ids
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    print('Data cached in {}_pickle file'.format(split_idx))


def get_pickle(model_cfg, split_idx):
    pickle_file = 'pickle/{}mer_{}.pickle'.format(model_cfg.kmer, split_idx)
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)  # 反序列化，与pickle.dump相反
        X = pickle_data['X']
        labels = pickle_data['labels']
        set_idxs = pickle_data['set_idxs']
        set_labels = pickle_data['set_labels']
        input_ids = pickle_data['input_ids']
        attention_masks = pickle_data['attention_masks']
        token_type_ids = pickle_data['token_type_ids']
        del pickle_data  # 释放内存
    print('Data and modules loaded.')

    return X, labels, set_idxs, set_labels, input_ids, attention_masks, token_type_ids

