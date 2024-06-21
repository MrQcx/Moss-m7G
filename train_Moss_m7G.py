#!/usr/bin/env Python
# coding=utf-8

from tqdm import tqdm
import itertools
import math
import numpy as np
import os
import random

import time
import torch
tqdm.pandas(ascii=True)
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
from models.model import Moss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda", 0)

def read_fasta(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.replace('\n', '')
            if line.startswith('>'):
                # label.append(int(line[-1]))
                if 'neg' in line:
                    label.append(0)
                else:
                    label.append(1)
            else:
                seq.append(line.replace('U', 'T'))

    return seq, label

def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    if overlap:
        max_length = max_seq - k + 1

    else:
        max_length = max_seq // k

    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict[seq[i:i+k]])

        encoded_sequences.append(encoded_seq+[0]*(max_length-len(encoded_seq)))

    return np.array(encoded_sequences)

def to_log(log, params):
    with open(f"results/train_diff_len_{params['seed']}_{params['seq_len']}.log", "a+") as f:
        f.write(log + '\n')

# ========================================================================================

def train_model(train_loader, valid_loader, test_loader, params):
    # Define model
    model = Moss(kernel_num=params['kernel_num'], topk=params['topk']).to(device)

    # Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    criterion_CE = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(params['epoch']):
        model.train()
        loss_ls = []
        t0 = time.time()
        for seq, label in train_loader:
            seq, label = seq.to(device), label.to(device)
            logits, _ = model(seq)
            loss = criterion_CE(logits, label)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())

        # Validation step (if needed)
        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_loader, model)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_loader, model)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        print(results)
        # to_log(results, params)

        if valid_acc > best_acc:
            best_acc = valid_acc
            test_performance, test_roc_data, test_prc_data = evaluate(test_loader, model)
            test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tAUC,\tPRE]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                test_performance[4], test_performance[5]) + '\n' + '=' * 60
            print(test_results)


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []

    for j, (data, labels) in enumerate(data_iter, 0):
        labels = labels.to(device)
        data = data.to(device)
        output, _ = net(data)

        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def evaluation_method(params):

    train_x, train_y = read_fasta('data/train.fasta')
    valid_x, valid_y = read_fasta('data/valid.fasta')
    test_x, test_y = read_fasta('data/test.fasta')


    seq_len = params['seq_len']
    train_x, train_y = np.array(train_x), np.array(train_y)
    valid_x, valid_y = np.array(valid_x), np.array(valid_y)
    test_x, test_y = np.array(test_x), np.array(test_y)

    train_x = encode_sequence_1mer(train_x, max_seq=seq_len)
    valid_x = encode_sequence_1mer(valid_x, max_seq=seq_len)
    test_x = encode_sequence_1mer(test_x, max_seq=seq_len)

    train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
    valid_dataset = TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
    test_dataset = TensorDataset(torch.tensor(test_x), torch.tensor(test_y))

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    train_model(train_loader, valid_loader, test_loader, params)


def main():

    params = {
        'kernel_num': 4096,
        'topk': 128,
        'lr': 0.0001,
        'batch_size': 128,
        'epoch': 100,
        'seq_len': 501,
        'saved_model_name': 'diff_len_',
        'seed': 17,
    }
    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    evaluation_method(params)

if __name__ == '__main__':
    main()