import pickle

from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
import matplotlib.patheffects as pe

from Model.loss import glorot
from utils.dataLoaders import Data
from utils.utils import move_to_device, squeeze_dim, adjust_learning_rate_cos
import math
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import random
import argparse
import numpy as np
import torch.nn as nn
from copy import deepcopy
from datetime import timedelta
from collections import OrderedDict
from timeit import default_timer as timer

from utils.logger import Logger
from utils.evaluation import compute_retrieval_precision, compute_hamming_distance, \
    compute_retrieval_precision_median_threshold, compute_median_threshold_binary_code_retrieval_precision, \
    compute_retrieval_precision_coarse, compute_retrieval_precision_q, evaluate_classification_accuracy, \
    evaluate_classification_accuracy_unsup, evaluate_pl


class unsupBaseModel_bert(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.test_loader = None
        self.val_loader = None
        self.database_loader = None
        self.train_loader = None
        self.data = None
        self.hparams = hparams
        self.high_confi_num = 0
        self.load_data()

    def load_data(self):
       # data =
        self.data = Data(self.hparams)
        self.train_loader, self.database_loader, self.val_loader, self.test_loader = self.data.get_loaders()



    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def prototype_visualization(self,is_coarse,label_num):
        raise NotImplementedError

    def configure_optimizers(self):
        #raise NotImplementedError
        return torch.optim.SGD(self.parameters(), lr = self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.wd)  # args.wd)

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        print("*" * 50)
        print(self.hparams.model_path + '.log')
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs
        self.display_meta_dataset(logger)
        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)

        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())

        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))
        val_perf, test_perf = self.run_test_coarse()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))



    def display_meta_dataset(self,logger):
        print(type(self.train_loader))
        n_tr = len(self.train_loader.dataset)
        n_va = len(self.val_loader.dataset)
        iter_per_epoch_tr = len(self.train_loader)
        iter_per_epoch_va = len(self.val_loader)

        logger.log('dataset={:s}'.format(self.hparams.dataset))
        print('training: size={:d},'.format(n_tr),
              'iter per epoch={:d} |'.format(iter_per_epoch_tr),
              'validation: size={:d},'.format(n_va),
              'iter per epoch={:d}'.format(iter_per_epoch_va))

    def adjust_learning_rate_cos(self,optimizer, lr, epoch, num_epochs, num_cycles):
        """Decay the learning rate based on schedule"""
        epochs_per_cycle = math.floor(num_epochs / num_cycles)
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch % epochs_per_cycle) / epochs_per_cycle))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return None

    def init_prototypes_with_cluster_center(self, fine_prototypes, coarse_prototypes,fine_nums,requires=True):
        raise NotImplementedError

    def update_prorotypes_assignment(self):
        raise NotImplementedError

    def cluster2getCoarsePrototype(self):

        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        # fine_centers = torch.tensor(glorot([self.hparams.fine_cluster_nums, self.hparams.encode_length])).to(device)
        fine_centers = glorot([self.hparams.fine_cluster_nums, self.hparams.encode_length]).clone().to(device)
        coarse_ids_x, coarse_centers = kmeans(X=fine_centers, num_clusters=self.hparams.n_class, distance='euclidean',
                                              device=device)
        coarse2fine_dict = {}  # coarse_labels:{fine_prototypes}
        for i in range(0, self.hparams.n_class):
            coarse2fine_dict.update({i: list()})
        for i in range(0, coarse_ids_x.shape[0]):
            coarse_index = (int)(coarse_ids_x[i])
            coarse2fine_dict.get(coarse_index).append(fine_centers[i].detach().cpu().numpy())
        rearranged_fine_prototypes = []
        fine_nums = []
        for key in coarse2fine_dict.keys():
            rearranged_fine_prototypes.extend(coarse2fine_dict.get(key))
            fine_nums.append(len(coarse2fine_dict.get(key)))
        rearranged_fine_prototypes = np.array(rearranged_fine_prototypes)
        rearranged_fine_prototypes = torch.tensor(rearranged_fine_prototypes).to(device).squeeze()
        coarse_centers = coarse_centers.to(device)
        return rearranged_fine_prototypes, coarse_centers, fine_nums

    def cluster2getCoarsePrototype_withmu(self):
        self.eval()
        with torch.no_grad():
            device = torch.device('cuda' if self.hparams.cuda else 'cpu')
            embed_chunks = []
            for (docs, _, _, _) in self.database_loader:
                # docs = squeeze_dim(move_to_device(docs ,device), dim=1)
                docs = docs.to(device)
                # forward = self.forward(inputs,labels_coarse, selected,q_new,device)

                embed_chunks.append(self.encode_projection(docs))
            #     embed_chunks.append(self.encode_Z_for_cluster(docs))

            data = torch.cat(embed_chunks, 0).to(device)
            data_size, dims, num_clusters = data.shape[0], self.hparams.encode_length, self.hparams.fine_cluster_nums
            # data = torch.from_numpy(data)
            # data = torch.tensor(embed_chunks).to(device)

            coarse_ids_x, coarse_centers = kmeans(X=data, num_clusters=self.hparams.n_class,
                                                  distance='euclidean',
                                                  device=device)

        fine_centers = glorot([self.hparams.fine_cluster_nums, self.hparams.encode_length]).clone().to(device)
        self.init_prototypes_with_cluster_center(fine_centers,coarse_centers.to(device),[0],True)
        self.update_prorotypes_assignment()

    def cluster2getPrototype(self):
        self.eval()
        with torch.no_grad():
            device = torch.device('cuda' if self.hparams.cuda else 'cpu')
            embed_chunks = []
            for (docs, _, _, _) in self.database_loader:
                # docs = squeeze_dim(move_to_device(docs ,device), dim=1)
                docs =docs.to(device)

                embed_chunks.append(self.encode_projection(docs))

            data = torch.cat(embed_chunks, 0).to(device)
            data_size, dims, num_clusters = data.shape[0], self.hparams.encode_length, self.hparams.fine_cluster_nums
            start =timer()
            fine_ids_x, fine_centers = kmeans(
            X=data, num_clusters=num_clusters, distance='euclidean', device=device
            )
            print("fine prototype cluster time:" + str(timedelta(seconds=round(timer() - start))))
            start = timer()
            coarse_ids_x,coarse_centers = kmeans(X=fine_centers,num_clusters=self.hparams.n_class,distance='euclidean',device=device)
            coarse2fine_dict={} # coarse_labels:{fine_prototypes}
            for i in range(0,self.hparams.n_class):
                coarse2fine_dict.update({i:list()})
            for i in range(0,coarse_ids_x.shape[0]):
                coarse_index =(int) (coarse_ids_x[i])
                coarse2fine_dict.get(coarse_index).append(fine_centers[i].detach().cpu().numpy())
            rearranged_fine_prototypes=[]
            fine_nums = []
            for key in coarse2fine_dict.keys():
                rearranged_fine_prototypes.extend(coarse2fine_dict.get(key))
                fine_nums.append(len(coarse2fine_dict.get(key)))
        # 得到每个样本所属的粗粒度中心
            for i in range(0,fine_ids_x.shape[0]):
                fine_ids_x[i] = coarse_ids_x[fine_ids_x[i]]
            print("coarse prototype cluster time:" + str(timedelta(seconds=round(timer() - start))))
            # print(type(self.train_loader.dataset.coarse_labels))
            # self.train_loader.dataset.coarse_labels = fine_ids_x.tolist()
            rearranged_fine_prototypes = np.array(rearranged_fine_prototypes)
            rearranged_fine_prototypes = torch.tensor(rearranged_fine_prototypes).to(device).squeeze()
            coarse_centers = coarse_centers.to(device)
        self.train()
        return rearranged_fine_prototypes,coarse_centers,fine_nums



    def renew_Y(self):
        self.eval()
        with torch.no_grad():
            device = torch.device('cuda' if self.hparams.cuda else 'cpu')
            embed_chunks = []
            coarse_label = []
            for (docs, _, _, _) in self.database_loader:
                # docs = squeeze_dim(move_to_device(docs ,device), dim=1)
                docs =docs.to(device)
                y = self.get_batchy(docs).cpu().detach().array().tolist()
                coarse_label.append(y)
            # forward = self.forward(inputs,labels_coarse, selected,q_new,device)
                embed_chunks.append(self.encode_projection(docs))
            #     embed_chunks.append(self.encode_Z_for_cluster(docs))

            self.train_loader.dataset.coarse_labels = coarse_label

        self.train()




    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            #print(self.hparams)
            for hparam, values in self.get_hparams_grid().items():
                #print(hparam)

                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)
        if not self.hparams.continue_training:
            self.define_parameters()
        else:
            print("continue training, using parameters from checkpoint")
        # logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())

        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr = self.hparams.lr)#self.configure_optimizers()
        # optimizer = self.configure_optimizers() #SGD+ cosine decay

        gradient_clippers = self.configure_gradient_clippers()
        best_val_perf = float('-inf')
        best_state_dict = None
        bad_epochs = 0
        step_2_on = False
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                self.high_confi_num = 0
                forward_sum = {}
                num_steps = 0
                if self.hparams.continue_training and epoch==1:
                    step_2_on = True
                    self.init_prototypes_with_original_cluster_center()
                elif (epoch==1 or epoch == self.hparams.pretrain_epoch) and not step_2_on:
                    fine_prototypes,coarse_prototypes,fine_nums = self.cluster2getPrototype()
                    self.init_prototypes_with_cluster_center(fine_prototypes,coarse_prototypes,fine_nums,True)
                    torch.save({'hparams': self.hparams,
                                'state_dict':self.state_dict()}, self.hparams.model_path+'pretrain')# checkpoint
                    best_val_perf = float('-inf')
                    best_state_dict = None
                    bad_epochs = 0
                else:
                    self.update_prorotypes_assignment()
                if epoch % 2 == 1 and epoch<10 and not step_2_on:
                    best_val_perf = float('-inf')
                    best_state_dict = None
                    bad_epochs = 0
                    fine_prototypes,coarse_prototypes,fine_nums = self.cluster2getPrototype()
                    self.init_prototypes_with_cluster_center(fine_prototypes,coarse_prototypes,fine_nums,True)
                fine_prototypes, coarse_prototypes, fine_nums = self.cluster2getPrototype()
                self.init_prototypes_with_cluster_center(fine_prototypes, coarse_prototypes, fine_nums, False)
                print(self.pq_head.Fine_books)
                for batch_num, batch in enumerate(self.train_loader):
                    optimizer.zero_grad()
                    bert_0,bert_1, labels_coarse, labels, selected= batch
                    bert_0 = bert_0.to(device)
                    bert_1 = bert_1.to(device)

                    if epoch < self.hparams.pretrain_epoch:
                        forward = self.forward_and_infer(bert_0, bert_1, device)
                    elif step_2_on:
                        # self-labeling module
                        forward = self.self_label_forward(bert_0,bert_1,device)
                    else:
                        # vae only
                        if self.hparams.VAE_only:
                            forward =self.forward_with_no_augmentation(bert_0,labels_coarse,device)

                        else:
                        # with contrastive
                            forward = self.forward_and_infer(bert_0, bert_1, device)

                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()
                    # print(self.pq_head.mu_s.grad)
                    # print(self.pq_head.mu_y.grad)
                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)
                    optimizer.step()

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break

                val_perf = self.evaluate(self.database_loader, self.val_loader, device,
                                         distance_metric=self.hparams.distance_metric,num_retrieve=self.hparams.num_retrieve)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.2f}'.format(
                    key, forward_sum[key] / num_steps)
                    for key in forward_sum]), False)
                logger.log(' | val perf {:8.2f}'.format(val_perf), False)
                print(self.high_confi_num)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs and not step_2_on:
                    torch.save({'hparams': self.hparams,
                                'state_dict': best_state_dict}, self.hparams.model_path + 'step1')  # 存个checkpoint
                    self.load_state_dict(best_state_dict)
                    self.hparams.lr = 0.000001
                    bad_epochs = 0
                    best_val_perf = float('-inf')
                    step_2_on = True
                elif bad_epochs> self.hparams.num_bad_epochs and step_2_on:
                    break
        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        return best_state_dict, best_val_perf


    def fine_assignment_evaluation(self,fine_num):
        device =  'cuda' if self.hparams.cuda else 'cpu'
        self.pq_head.fine_assign_dict = {}
        for i in range(fine_num):
            self.pq_head.fine_assign_dict.update({str(i):0})
        for (docs, labels_coarse, _, _) in self.train_loader:
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            self.pq_fine_assignment_evaluation(docs,labels_coarse,fine_num)
        for i in range(fine_num):
            print(str(i) + ':' + str(self.pq_head.fine_assign_dict.get(str(i))))
        self.train()

    def pq_fine_assignment(self,bert0,is_coarse):
        raise NotImplementedError

    def evaluate(self,database_loader, eval_loader, device, distance_metric, num_retrieve, is_median=False,is_coarse=False):
        self.eval()
        with torch.no_grad():
            if is_median:
                print("median")
                perf = compute_median_threshold_binary_code_retrieval_precision(database_loader, eval_loader, device,
                                                                                self.encode_projection, distance_metric,
                                                                                num_retrieve)
            elif is_coarse:
                perf = compute_retrieval_precision_coarse(database_loader, eval_loader,
                                                  device, self.encode_discrete,  # prob_discrete_for_training
                                                  distance_metric,
                                                  num_retrieve)
            else:
                perf = compute_retrieval_precision(database_loader, eval_loader,
                                                   device, self.encode_discrete,  # prob_discrete_for_training
                                                   distance_metric,
                                                   num_retrieve)

        self.train()
        return perf


    def run_test_coarse(self):
        start = timer()
        device = 'cuda' if self.hparams.cuda else 'cpu'
        print("coarse test result")
        val_perf = self.evaluate(self.database_loader, self.val_loader, device, self.hparams.distance_metric, self.hparams.num_retrieve, self.hparams.median,
                                 True)
        print("val time:"+str(timedelta(seconds=round(timer() - start))))
        val = timer()
        test_perf = self.evaluate(self.database_loader, self.test_loader, device, self.hparams.distance_metric, self.hparams.num_retrieve, self.hparams.median,
                                  True)
        print("test time:"+str(timedelta(seconds=round(timer() - val))))
        print(("whole time:"+str(timedelta(seconds=round(timer() - start)))))
        return val_perf, test_perf

    def run_test(self):
        start = timer()
        device = 'cuda' if self.hparams.cuda else 'cpu'
        val_perf = self.evaluate(self.database_loader, self.val_loader, device, self.hparams.distance_metric,
                                 self.hparams.num_retrieve, self.hparams.median,
                                 self.hparams.is_coarse)
        print("val time:" + str(timedelta(seconds=round(timer() - start))))
        val = timer()
        test_perf = self.evaluate(self.database_loader, self.test_loader, device, self.hparams.distance_metric,
                                  self.hparams.num_retrieve, self.hparams.median,
                                  self.hparams.is_coarse)
        print("test time:" + str(timedelta(seconds=round(timer() - val))))
        print(("whole time:" + str(timedelta(seconds=round(timer() - start)))))
        return val_perf, test_perf

    def hash_code_visualization_coarse(self,is_coarse,label_num):
        self.eval()
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        database_loader = self.database_loader
        retrievalB = list([])
        retrievalL = list([])
        coarse = list([])
        for batch_step, batch in enumerate(database_loader):
            # inputs, labels_coarse, labels, selected = batch
            docs = batch[0]
            # coarse
            # target = batch[1]
            # fine
            target_coarse = batch[1]
            coarse.extend(target_coarse.cpu().data.numpy())
            target = batch[2]
            # print(batch[1].shape)
            # target = batch[index_of_labels]#.argmax(dim=1)
            # print(batch[0].shape)
            docs = squeeze_dim(move_to_device(docs, device), dim=1)
            #docs = docs.to(device)

            code = self.encode_discrete(docs)
            # code = docs
            retrievalB.extend(code.cpu().data.numpy())
            retrievalL.extend(target.cpu().data.numpy())
       # for (docs, labels, _, _) in loader:
        hash_codes = np.array(retrievalB)
        labels = np.array(retrievalL)
        labels_coarse = np.array(coarse,dtype=int)
        print(labels.shape)
        # print(labels.astype(np.int).shape)
        print(hash_codes.shape)
        choose_fine = 7
        print(labels[labels_coarse==choose_fine].shape)
        pickle.dump((hash_codes, labels), open('coarse_hash_codes_{:d}bits.pk'.format(self.hparams.encode_length), 'wb'))
        # TSN
        mapper = TSNE(perplexity=100,random_state=10001).fit_transform(hash_codes)
        print(mapper[labels_coarse == choose_fine,:].shape)
        print(mapper.shape)
        plt.figure(figsize=(8.5, 8))
        plt.scatter(mapper[:, 0], mapper[:, 1], lw=0, s=20, c=labels[:].astype(np.int), cmap='Spectral')
        #fine
        # plt.scatter(mapper[labels_coarse==choose_fine, 0], mapper[labels_coarse==choose_fine, 1], lw=0, s=20, c=labels[labels_coarse==choose_fine].astype(np.int), cmap='Spectral')
        # plt.scatter(mapper[0, 0], mapper[0, 1], lw=0, s=100,marker='p', c='k')
        # plt.scatter(mapper[9:, 0], mapper[9:, 1], lw=0, s=20, c=labels[9:].astype(np.int), cmap='Spectral')
        # plt.scatter(mapper[0:9, 0], mapper[0:9, 1], lw=0, s=100, marker='p', c=labels[0:9])

        plt.axis("off")
        plt.gcf().tight_layout()
        plt.savefig('coarse_DB_{:d}bits.pdf'.format(self.hparams.encode_length), bbox_inches='tight',
                    pad_inches=0.0)


    def run_coarse_test(self,is_coarse):
        start = timer()
        device = 'cuda' if self.hparams.cuda else 'cpu'
        val_perf = self.evaluate(self.database_loader, self.val_loader, device, self.hparams.distance_metric,
                                 self.hparams.num_retrieve, self.hparams.median,
                                 is_coarse)
        print("val time:" + str(timedelta(seconds=round(timer() - start))))
        val = timer()
        test_perf = self.evaluate(self.database_loader, self.test_loader, device, self.hparams.distance_metric,
                                  self.hparams.num_retrieve, self.hparams.median,
                                  is_coarse)
        print("test time:" + str(timedelta(seconds=round(timer() - val))))
        print(("whole time:" + str(timedelta(seconds=round(timer() - start)))))
        return val_perf, test_perf

    def run_coarse_topN(self,is_coarse=True,N=1000):

        device = 'cuda' if self.hparams.cuda else 'cpu'
        topN_result = []
        self.eval()
        with torch.no_grad():
          for i in range(50,N,50):
                perf = compute_retrieval_precision_coarse(self.database_loader, self.test_loader,
                                                          device, self.encode_discrete,  # prob_discrete_for_training
                                                          self.hparams.distance_metric,
                                                          i)
                topN_result.append(perf)
                print(perf)
        with open('./'+self.hparams.data_path+'coarse_ours.txt','w+') as f:
            for i in range(0,len(topN_result)):
                f.write(str(float(topN_result[i]))+'\n')
        f.close()



    def load(self,hparams=None):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.cuda \
            else torch.load(self.hparams.model_path,
                            map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        if hparams == None:
            self.hparams = checkpoint['hparams']
        else:
            self.hparams = hparams
        self.define_parameters()
        # print('Loaded model with: %s' %self.flag_hparams())
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)

    def flag_hparams(self):
        flags = '%s %s' % (self.hparams.model_path, self.hparams.data_path)
        for hparam in vars(self.hparams):
            #   getattr(x, 'y') is equivalent to x.y.
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'data_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.001, 0.005, 0.0001, 1e-5, 1e-6],
            'batch_size': [32, 64, 128, 256, 512]
        })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('model_path', type=str,help='path to save the model')
        parser.add_argument('--data_path', default='./data_WOS',metavar='DIR', help='path to dataset')
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--continue_training', action='store_true',
                            help='continue to train a model?')
        parser.add_argument('--dataset', default='WOS', choices=['WOS', '20news','nyt','yelp','arxiv','BERT'])
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                                 '[%(default)d]')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed [%(default)d]')
        parser.add_argument("--batch_size", default=512, type=int,
                            help='batch size [%(default)d]')
        parser.add_argument('--epochs', type=int, default=100,
                            help='max number of epochs [%(default)d]')
        parser.add_argument("--lr", default=1e-4, type=float,
                            help='initial learning rate [%(default)g]')
        parser.add_argument("-l", "--encode_length", type=int, default=32,
                            help="Number of bits of the hash code [%(default)d]")
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')
        parser.add_argument('--workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--max_length', type=int, default=200,
                            help='max number of sentence length [%(default)d]')
        parser.add_argument('--wd', default=0.0001, type=float, metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')  # 暂时没用
        parser.add_argument('--hiddim', default=128, type=int, help='decoder dimension')
        parser.add_argument('--distance_metric', default='hamming',
                            choices=['hamming', 'cosine']),
        parser.add_argument('--num_retrieve', type=int, default=100,
                            help='num neighbors to retrieve [%(default)d]')
        parser.add_argument('--is_coarse', action='store_true',
                            help='use coarse class to get precision(when testing)')
        parser.add_argument('--c2f', action='store_true',
                            help='use coarse class label to train')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')
        parser.add_argument('--median', action='store_true',
                            help='median threshold (VDSH) ?')
        parser.add_argument('--mlp', action='store_false', help='use mlp head')
        parser.add_argument('--tau', default=0.1, type=float, help="Temperature [%(default)d]")
        parser.add_argument('--n-class', default=7, type=int,
                            help='the number of superclasses, e.g., 7 for WOS, 7 for 20news')# 3:yelp&arxiv;5:nyt
        parser.add_argument('--pretrain_epoch', default=10, type=int,
                            help='the number of instance discrimination pretrain')
        parser.add_argument('--fine_cluster_nums', default=70, type=int,
                            help='the number of fine clusters')  # 3:yelp&arxiv;5:nyt
        parser.add_argument('--VAE_only', action="store_true")

        return parser

