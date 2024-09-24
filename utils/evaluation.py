import numpy as np
import torch
import torch.nn as nn

from utils.utils import squeeze_dim, move_to_device
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from termcolor import colored
## compute top-100 precision for coarse class
def compute_retrieval_precision_coarse(train_loader, eval_loader, device,
                                encode_discrete=None, distance_metric='hamming',
                                num_retrieve=100):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        # for (docs, _,labels,_) in loader: # x, coarse, fine, index
        for (docs, labels, _, _) in loader:
            # docs =torch.squeeze(docs.to(device),dim=1)
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks.append(docs if encode_discrete is None else  # encode_discrete is a function
                                   encode_discrete(docs))
            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    src_encodings, src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)
    mid_val = torch.median(src_encodings,dim = 0)

    # prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
    #                                       src_encodings, src_label_lists,
    #                                       num_retrieve, distance_metric)
    # print(prec)
    src_label_lists = torch.tensor(src_label_lists).to(device)
    # # # tgt_encodings = torch.tensor(tgt_encodings).to(device)
    tgt_label_lists = torch.tensor(tgt_label_lists).to(device)
    # #
    if distance_metric == 'hamming':
        prec = evaluate_retrieval_accuracy(src_encodings, src_label_lists, tgt_encodings, tgt_label_lists, num_retrieve)
    else:
        prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                              src_encodings, src_label_lists,
                                              num_retrieve, distance_metric)
    return prec

def compute_retrieval_precision_q(train_loader, eval_loader, device,
                                encode_discrete=None, encode_discrete_q=None,distance_metric='hamming',
                                num_retrieve=100):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, _,labels,_) in loader: # x, coarse, fine, index
            # docs =torch.squeeze(docs.to(device),dim=1)
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks.append(docs if encode_discrete is None else # encode_discrete is a function
                                   encode_discrete(docs))

            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)

        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    def extract_data_q(loader):

        encoding_chunks_q = []
        label_chunks = []
        for (docs, _,labels,_) in loader: # x, coarse, fine, index
            # docs =torch.squeeze(docs.to(device),dim=1)
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks_q.append(docs if encode_discrete_q is None else  # encode_discrete is a function
                                   encode_discrete_q(docs))
            label_chunks.append(labels)

        encoding_mat_q = torch.cat(encoding_chunks_q,0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat_q,label_lists

    src_encodings, src_label_lists = extract_data_q(train_loader)
    tgt_encodings, tgt_label_lists = extract_data_q(eval_loader)

    #
    # prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
    #                                       src_encodings, src_label_lists,
    #                                       num_retrieve, distance_metric)

    # # src_encodings = torch.tensor(src_encodings).to(device)
    src_label_lists = torch.tensor(src_label_lists).to(device)
    # # # tgt_encodings = torch.tensor(tgt_encodings).to(device)
    tgt_label_lists = torch.tensor(tgt_label_lists).to(device)
    # #
    prec,top_k_indices = evaluate_retrieval_accuracy_q(src_encodings, src_label_lists, tgt_encodings, tgt_label_lists, num_retrieve)
    print(prec)
    del src_encodings
    del tgt_encodings
    del src_label_lists
    del tgt_label_lists
    src_encodings,src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)
    src_label_lists = torch.tensor(src_label_lists).to(device)
    tgt_label_lists = torch.tensor(tgt_label_lists).to(device)
    src_encodings =src_encodings[top_k_indices]
    src_encodings = src_encodings.squeeze()
    src_label_lists = src_label_lists[top_k_indices].squeeze()
    prec = evaluate_retrieval_accuracy_q_2(src_encodings, src_label_lists, tgt_encodings, tgt_label_lists, num_retrieve)
    del src_encodings
    del tgt_encodings
    del src_label_lists
    del tgt_label_lists

    return prec

from sklearn.metrics import accuracy_score
def evaluate_classification_accuracy(test_loader,device,pred_class,is_coarse):
    def extract_data(loader,is_coarse=False):
        pred_chunks = []
        label_chunks = []
        if is_coarse:
            for (docs, labels, _, _) in loader:  # x, coarse, fine, index
                # docs =torch.squeeze(docs.to(device),dim=1)
                docs = squeeze_dim(move_to_device(docs, device), dim=1)
                pred_chunks.append(docs if pred_class is None else  # encode_discrete is a function
                                   pred_class(docs, is_coarse))
                label_chunks.append(labels)
        else:

            for (docs, c_labels, labels, _) in loader:  # x, coarse, fine, index
                # docs =torch.squeeze(docs.to(device),dim=1)
                docs = squeeze_dim(move_to_device(docs, device), dim=1)
                pred_chunks.append(docs if pred_class is None else  # encode_discrete is a function
                                   pred_class(docs, is_coarse,c_labels))
                label_chunks.append(labels)



        encoding_mat = torch.cat(pred_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    pred, target = extract_data(test_loader,is_coarse)
    pred = torch.tensor(pred).to(device)
    target = torch.tensor(target).to(device)
    if not is_coarse:
        pred_dict, pred_stats = evaluate_pl(target,pred)
        print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
            pred_dict.items(), key=lambda x: x[0])]))
        print(colored(''.join([" | {}: {:.4f}".format(k, v)
                           for k, v in pred_stats.items()]), 'yellow'))
        acc =pred_dict.get('ACC')
    else:
        pred = pred.argmax(dim=1)
        acc = torch.eq(pred,target.squeeze(dim=-1)).float().mean()

    # acc = accuracy_score(pred,target)
    return acc

def evaluate_classification_accuracy_unsup(test_loader,device,pred_class,is_coarse):
    def extract_data(loader,is_coarse=False):
        pred_chunks = []
        label_chunks = []
        if is_coarse:
            # for (docs, labels, _, _) in loader:  # x, coarse, fine, index
            for (docs, _, labels, _) in loader:  # x, coarse, fine, index
                # docs =torch.squeeze(docs.to(device),dim=1)
                docs = squeeze_dim(move_to_device(docs, device), dim=1)
                pred_chunks.append(docs if pred_class is None else  # encode_discrete is a function
                                   pred_class(docs, is_coarse))
                label_chunks.append(labels)
        else:

            for (docs, c_labels, labels, _) in loader:  # x, coarse, fine, index
                # docs =torch.squeeze(docs.to(device),dim=1)
                docs = squeeze_dim(move_to_device(docs, device), dim=1)
                pred_chunks.append(docs if pred_class is None else  # encode_discrete is a function
                                   pred_class(docs, is_coarse))
                label_chunks.append(labels)



        encoding_mat = torch.cat(pred_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    pred, target = extract_data(test_loader,is_coarse)
    pred = torch.tensor(pred).to(device)
    target = torch.tensor(target).to(device)
    pred_dict, pred_stats = evaluate_pl(target, pred)
    print(", ".join(["{}: {}".format(k, v) for k, v in sorted(
        pred_dict.items(), key=lambda x: x[0])]))
    print(colored(''.join([" | {}: {:.4f}".format(k, v)
                           for k, v in pred_stats.items()]), 'yellow'))
    acc = pred_dict.get('ACC')

    # acc = accuracy_score(pred,target)
    return acc

def evaluate_pl(target,prob,prob_one_hot = False):
    # num_classes = prob.shape[1]
    # device = target.device
    # target = target.detach().cpu().numpy()
    # target = target.astype(np.int64)
    # targets = np.zeros([len(target), num_classes])
    # targets[np.arange(len(target)), target] = 1
    # targets = torch.tensor(targets, dtype=torch.float32)  # (n, num_class)
    # targets = targets.to(device)
    target = target.squeeze()
    # pred_class_num = prob.shape[1]
    pred_class_num = torch.unique(target).numel()

    if prob_one_hot:
        pred = prob
    else:
        pred = prob.argmax(dim=1)
        pred = pred.squeeze()
    if prob_one_hot:
        clustering_stats,match = hungarian_evaluate_target(pred,target)
    else:
        clustering_stats, match = hungarian_evaluate(
            pred,target,prob,pred_class_num)
    pred, counts = pred.unique(
            return_counts=True)
    reordered_preds = torch.zeros(len(pred), dtype=pred.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[pred == int(pred_i)] = int(target_i)
    pred_dict = dict(
            zip(reordered_preds.cpu().tolist(), counts.tolist()))
    return pred_dict, clustering_stats

def hungarian_evaluate(predictions,targets,probs, pred_class_num,multilabel=False):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching

    if multilabel:
        num_classes = targets.shape[1]
    else:
        num_classes = torch.unique(targets).numel()
        # num_classes = probs.shape[1]
    num_elems = targets.size(0)
    print(pred_class_num)
    print(num_classes)
    match = _hungarian_match(predictions, targets,
                             preds_k=pred_class_num, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    if multilabel:
        reordered_preds = F.one_hot(reordered_preds, targets.shape[1])
        acc = int(((reordered_preds * targets).sum(dim=1)
                  > 0).sum()) / float(num_elems)
        ari = np.NaN
        nmi = np.NaN
        top5 = np.NaN

    else:
        # Gather performance metrics
        acc = int((reordered_preds == targets).sum()) / float(num_elems)
        nmi = metrics.normalized_mutual_info_score(
            targets.cpu().numpy(), predictions.cpu().numpy())
        ari = metrics.adjusted_rand_score(
            targets.cpu().numpy(), predictions.cpu().numpy())

        _, preds_top5 = probs.topk(2, 1, largest=True)
        reordered_preds_top5 = torch.zeros_like(preds_top5)
        for pred_i, target_i in match:
            reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
        correct_top5_binary = reordered_preds_top5.eq(
            targets.view(-1, 1).expand_as(reordered_preds_top5))
        top5 = float(correct_top5_binary.sum()) / float(num_elems)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5}, match
    # return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}

def hungarian_evaluate_target(predictions,targets):
    # 直接匈牙利投票，没有概率向量
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching


    num_classes = torch.unique(targets).numel()
        # num_classes = probs.shape[1]
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets,
                             preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(
        targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(
        targets.cpu().numpy(), predictions.cpu().numpy())

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi}, match


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    # assert (preds_k == targets_k)  # one to one
    # num_k = preds_k
    num_correct = np.zeros((preds_k, targets_k))

    for c1 in range(preds_k):
        for c2 in range(targets_k):
            if flat_targets.dim() > 1:
                # elementwise, so each sample contributes once
                votes = int(
                    ((flat_preds == c1) * (flat_targets[:, c2] == 1)).sum())
                num_correct[c1, c2] = votes
            else:
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

## compute top-100 precision for fine class
def compute_retrieval_precision(train_loader, eval_loader, device,
                                encode_discrete=None, distance_metric='hamming',
                                num_retrieve=100):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, _,labels,_) in loader: # x, coarse, fine, index
            # docs =torch.squeeze(docs.to(device),dim=1)
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks.append(docs if encode_discrete is None else # encode_discrete is a function
                                   encode_discrete(docs))
            # encoding_chunks.append(docs)
            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    src_encodings, src_label_lists = extract_data(train_loader)
    tgt_encodings, tgt_label_lists = extract_data(eval_loader)

    #
    # prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
    #                                       src_encodings, src_label_lists,
    #                                       num_retrieve, distance_metric)
    # print(prec)
    # # src_encodings = torch.tensor(src_encodings).to(device)
    src_label_lists = torch.tensor(src_label_lists).to(device)
    # # # tgt_encodings = torch.tensor(tgt_encodings).to(device)
    tgt_label_lists = torch.tensor(tgt_label_lists).to(device)
    # #
    if distance_metric=='hamming':
        prec = evaluate_retrieval_accuracy(src_encodings, src_label_lists, tgt_encodings, tgt_label_lists, num_retrieve)
    else:
        prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric)
    return prec


def evaluate_retrieval_accuracy_q(train_code,  train_label, eval_code,  eval_label, num_retrieve=100):
    #print('new eval')
    K = min(num_retrieve, train_code.shape[0])
    assert train_code.shape[1] == eval_code.shape[1]

    topk_indices = []
    train_code = train_code
    chunk_size = 1500
    doc_indices = []
    # for i in range(0, eval_code_q.size(0), chunk_size):
    #     chunk_code = eval_code_q[i: i + chunk_size]
    #     chunk_scores = torch.mm(chunk_code, (1 - train_code.T)) + torch.mm(1 - chunk_code, train_code.T)
    #     _, chunk_indexes = torch.topk(chunk_scores, k=K*10, dim=1, largest=False)
    #
    #     topk_indices.append(train_label[chunk_indexes.cpu()])
    # topk_indices = torch.cat(topk_indices, dim=0)
    # accuracy = torch.sum(torch.eq(eval_label.unsqueeze(1), topk_indices)) / K * 100 / eval_code.shape[0]
    # print(accuracy)
    # train_code = train_code[topk_indices]
    for i in range(0, eval_code.size(0), chunk_size):
        chunk_code = eval_code[i: i + chunk_size]
        chunk_scores = torch.mm(chunk_code, (1 - train_code.T)) + torch.mm(1 - chunk_code, train_code.T)
        _, chunk_indexes = torch.topk(chunk_scores, k=K*2, dim=1, largest=False)
        doc_indices.append(chunk_indexes)

        topk_indices.append(train_label[chunk_indexes.cpu()])
    topk_indices = torch.cat(topk_indices, dim=0)
    doc_indices = torch.cat(doc_indices,dim=0)
    print(topk_indices.shape)
    print(eval_label.shape)
    # print(eval_code.shape[0])
    # precision = len([_ for candidates in candidate_lists
    #                  if not gold_set.isdisjoint(candidates)]) / K * 100
    accuracy = torch.sum(torch.eq(eval_label.unsqueeze(1),topk_indices)) / K * 50 / eval_code.shape[0]
    print(accuracy)
    # accuracy = torch.sum(torch.eq(eval_label, topk_indices.squeeze())) / K * 10 / eval_code.shape[0]
    # print(accuracy)
    # accuracy = torch.sum(torch.sum(eval_label.unsqueeze(1)*topk_indices, dim=-1).sign())/K*100/eval_code.shape[0]
    del topk_indices
    return accuracy,doc_indices



def evaluate_retrieval_accuracy_q_2(train_code,  train_label, eval_code,  eval_label, num_retrieve=100):
    #print('new eval')
    K = min(num_retrieve, train_code.shape[0])
    assert train_code.shape[0] == eval_code.shape[0]
    assert train_code.shape[2] == eval_code.shape[1]

    topk_indices = []

    chunk_size = 1500
    # for i in range(0, eval_code_q.size(0), chunk_size):
    #     chunk_code = eval_code_q[i: i + chunk_size]
    #     chunk_scores = torch.mm(chunk_code, (1 - train_code.T)) + torch.mm(1 - chunk_code, train_code.T)
    #     _, chunk_indexes = torch.topk(chunk_scores, k=K*10, dim=1, largest=False)
    #
    #     topk_indices.append(train_label[chunk_indexes.cpu()])
    # topk_indices = torch.cat(topk_indices, dim=0)
    # accuracy = torch.sum(torch.eq(eval_label.unsqueeze(1), topk_indices)) / K * 100 / eval_code.shape[0]
    # print(accuracy)
    # train_code = train_code[topk_indices]
    for i in range(0, eval_code.size(0), chunk_size):
        chunk_code = eval_code[i: i + chunk_size].unsqueeze(1)
        train_code = train_code[i: i + chunk_size].permute(0,2,1)
        chunk_scores = torch.matmul(chunk_code, (1 - train_code)) + torch.matmul(1 - chunk_code, train_code)
        _, chunk_indexes = torch.topk(chunk_scores.squeeze(), k=K, dim=1, largest=False)
        print(torch.gather(train_label,1,chunk_indexes).shape)
        topk_indices.append(torch.gather(train_label,1,chunk_indexes))
        # print(train_label[chunk_indexes.cpu()].shape)
        # print(train_label[:,chunk_indexes.cpu()])
        # print(train_label[chunk_indexes.cpu().unsqueeze(1)].shape)
        # print(train_label[chunk_indexes.cpu().unsqueeze(2)].shape)
        # topk_indices.append(train_label[chunk_indexes.cpu()])
    topk_indices = torch.cat(topk_indices, dim=0)
    print(topk_indices.shape)
    # print(eval_label.shape)
    # print(eval_code.shape[0])
    # precision = len([_ for candidates in candidate_lists
    #                  if not gold_set.isdisjoint(candidates)]) / K * 100
    accuracy = torch.sum(torch.eq(eval_label.unsqueeze(1),topk_indices.unsqueeze(2))) / K * 100 / eval_code.shape[0]
    print( torch.sum(torch.eq(eval_label.unsqueeze(1),topk_indices.unsqueeze(2)))/ eval_code.shape[0] )
    print(accuracy)
    # accuracy = torch.sum(torch.sum(eval_label.unsqueeze(1)*topk_indices, dim=-1).sign())/K*100/eval_code.shape[0]
    return accuracy,topk_indices


def evaluate_retrieval_accuracy(train_code, train_label, eval_code, eval_label, num_retrieve=100):
    #print('new eval')
    K = min(num_retrieve, train_code.shape[0])
    assert train_code.shape[1] == eval_code.shape[1]

    topk_indices = []
    train_code = train_code
    chunk_size = 1500
    for i in range(0, eval_code.size(0), chunk_size):
        chunk_code = eval_code[i: i + chunk_size]
        chunk_scores = torch.mm(chunk_code, (1 - train_code.T)) + torch.mm(1 - chunk_code, train_code.T)
        _, chunk_indexes = torch.topk(chunk_scores, k=K, dim=1, largest=False)

        topk_indices.append(train_label[chunk_indexes.cpu()])
    topk_indices = torch.cat(topk_indices, dim=0)
    # print(topk_indices.shape)
    # print(eval_label.shape)
    # print(eval_code.shape[0])
    # precision = len([_ for candidates in candidate_lists
    #                  if not gold_set.isdisjoint(candidates)]) / K * 100
    accuracy = torch.sum(torch.eq(eval_label.unsqueeze(1),topk_indices)) / K * 100 / eval_code.shape[0]
    # print(accuracy)
    # accuracy = torch.sum(torch.sum(eval_label.unsqueeze(1)*topk_indices, dim=-1).sign())/K*100/eval_code.shape[0]
    return accuracy


def compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                   src_encodings, src_label_lists,
                                   num_retrieve, distance_metric='hamming',
                                   chunk_size=100, binary=True):
    K = min(num_retrieve, len(src_encodings))
    D = compute_distance(tgt_encodings, src_encodings, distance_metric,
                         chunk_size, binary)

    # Random here in breaking ties (e.g., may have many 0-distance neighbors),
    # but given nontrivial representations this is not an issue (hopefully).
    #
    # TODO: maybe use a stable version of topk when available,
    #   https://github.com/pytorch/pytorch/issues/27542
    _, list_topK_nearest_indices = D.topk(K, dim=1, largest=False)

    average_precision = 0.
    for i, topK_nearest_indices in enumerate(list_topK_nearest_indices):
        gold_set = set(tgt_label_lists[i])
        candidate_lists = [src_label_lists[j] for j in topK_nearest_indices]
        precision = len([_ for candidates in candidate_lists
                         if not gold_set.isdisjoint(candidates)]) / K * 100
        average_precision += precision / tgt_encodings.size(0)

    return average_precision


def compute_distance(X1, X2, distance_metric='hamming', chunk_size=1000,
                     binary=True):
    if distance_metric == 'hamming':
        D = compute_hamming_distance(X1, X2, chunk_size=chunk_size,
                                     binary=binary)
    elif distance_metric == 'cosine':
        D = cosine_distance_torch(X1, X2)
    else:
        raise Exception('Unsupported distance: {0}'.format(distance_metric))
    return D


def compute_hamming_distance(X1, X2, chunk_size=100, binary=True):
    assert X1.size(1) == X2.size(1)
    N, m = X1.shape
    M, m = X2.shape

    D = []
    for i in range(0, X1.size(0), chunk_size):
        X1_chunk = X1[i:i + chunk_size]
        if binary:
            A = (1 - X1_chunk).float() @ X2.t().float()  # X2 one, X1_chunk zero
            B = X1_chunk.float() @ (1 - X2).t().float()  # X1_chunk one, X2 zero
            D.append(A + B)
        else:
            n = X1_chunk.shape[0]
            # Warning: This is extremely memory-intensive.
            D.append((X1_chunk.unsqueeze(1).expand(n, M, m) != X2).sum(dim=-1))

    return torch.cat(D, dim=0)  # N x M


# Copied from https://discuss.pytorch.org/t/pairwise-cosine-distance/30961/4.
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def compute_retrieval_precision_median_threshold(train_b, test_b, train_y, test_y, distance_metric='hamming',
                                                 num_retrieve=100):
    src_encodings = train_b
    src_label_lists = train_y
    tgt_encodings = test_b
    tgt_label_lists = test_y

    prec = compute_topK_average_precision(tgt_encodings, tgt_label_lists,
                                          src_encodings, src_label_lists,
                                          num_retrieve, distance_metric)
    return prec


def compute_median_threshold_binary_code_retrieval_precision(database_loader, eval_loader, device, encode_continuous=None, distance_metric='hamming',
                                num_retrieve=100):
    def extract_data(loader):
        encoding_chunks = []
        label_chunks = []
        for (docs, _,labels,_) in loader:
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            encoding_chunks.append(docs if encode_continuous is None else
                                   encode_continuous(docs))
            label_chunks.append(labels)

        encoding_mat = torch.cat(encoding_chunks, 0)
        label_mat = torch.cat(label_chunks, 0)
        label_lists = [[label_mat[i].item()] for i in range(label_mat.size(0))]
        return encoding_mat, label_lists

    src_rep, src_label = extract_data(database_loader)
    tgt_rep, tgt_label = extract_data(eval_loader)

    mid_val, _ = torch.median(src_rep, dim=0)

    src_encodings = (src_rep > mid_val).type(torch.FloatTensor).to(device)
    tgt_encodings = (tgt_rep > mid_val).type(torch.FloatTensor).to(device)
    src_label = torch.tensor(src_label).to(device)
    tgt_label = torch.tensor(tgt_label).to(device)
    # prec = compute_topK_average_precision(tgt_encodings, tgt_label,
    #                                       src_encodings, src_label,
    #                                       num_retrieve, distance_metric)
    prec = evaluate_retrieval_accuracy(src_encodings, src_label, tgt_encodings, tgt_label, num_retrieve)
    return prec