import pickle
import matplotlib.patheffects as pe
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.autograd import Function
from Model.loss import NtXentLoss, glorot, NtXentLoss_angular, y_glorot, anchor_glorot
import numpy as np
import math

from Model.unsupBaseModel_bert import unsupBaseModel_bert
from utils.utils import get_init_function, squeeze_dim, move_to_device

class VAE_head(nn.Module):
    def __init__(self,  encode_length, coarse_num,gumbel_temp, dist_metric, fine_Nums,proto_tau,yz_tau,consis_temp,sample_method="gumbel_softmax"):
        super(VAE_head, self).__init__()
        self.projectionHead = nn.Sequential(
            nn.Linear( encode_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, encode_length),
        )  # mu_head
        self.sigma_head = nn.Sequential(
            nn.Linear( encode_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, encode_length),
        )  # sigma_head
        self.apply(get_init_function(0.05))
        self.encode_length = encode_length
        self.gumbel_temp = gumbel_temp
        self.consis_temp = consis_temp
        self.dist_metric = dist_metric
        self.criterion_cls = nn.CrossEntropyLoss()
        self.Fine_books = nn.Parameter(torch.zeros(coarse_num), requires_grad=False)
        self.mu_y = nn.Parameter(glorot([coarse_num, encode_length]), requires_grad=True)
        self.mu_s = nn.Parameter(glorot([fine_Nums, encode_length]), requires_grad=True)
        self.coarse_num = coarse_num
        self.proto_tau = proto_tau
        self.yz_tau = yz_tau
        self.sample_method = sample_method

    def clusrered_prototype_init(self,fine_prototypes,coarse_prototypes,fine_nums,requires=True):
        self.Fine_books = nn.Parameter(torch.tensor(np.array(fine_nums)).to(self.mu_s.device),requires_grad=False)
        self.mu_s = nn.Parameter(fine_prototypes,requires_grad=requires)
        self.mu_y = nn.Parameter(coarse_prototypes,requires_grad=True)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature=1):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=1, hard=True):
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y
        raise NotImplementedError()

    def defined_sim(self, x, c):
        if self.dist_metric == "euclidean":
            diff = x.unsqueeze(1) - c.unsqueeze(0) # (bath_size, 1,L_word),(1,N_words, L_word)
            # diff:(batch_size,N_words,L_words)
            return - torch.sum(diff * diff, - 1) #(batch_size,N_words)
        else:
            raise NotImplementedError

    def defined_sim_cos(self, x, c):
        x = F.normalize(x,dim=1,p=2)
        c = F.normalize(c,dim=1,p=2)
        if self.dist_metric == "euclidean":
            diff = x.unsqueeze(1) @ c.t().unsqueeze(0) # (bath_size, 1,L_word),(1, L_word,N_words)=(batch,1,N_words)
            return diff.squeeze(1)
        else:
            raise NotImplementedError


    def get_hard_prob(self,soft_prob,i):
        hard_prob = torch.argmax(soft_prob, dim=1)
        device = hard_prob.device
        hard_prob = hard_prob.detach().cpu().numpy()
        hard_prob = hard_prob.astype(np.int64)
        batch_hard = np.zeros([len(hard_prob), int(self.Fine_books[i])])
        batch_hard[np.arange(len(hard_prob)), hard_prob] = 1
        batch_hard = torch.tensor(batch_hard, dtype=torch.float32)  # (n, num_class)
        batch_hard = batch_hard.to(device)
        return batch_hard

    def get_hard_prob_in_all_Fine_books(self,soft_prob):
        hard_prob = torch.argmax(soft_prob, dim=1)
        device = hard_prob.device
        hard_prob = hard_prob.detach().cpu().numpy()
        hard_prob = hard_prob.astype(np.int64)
        batch_hard = np.zeros([len(hard_prob), self.mu_s.shape[0]])
        batch_hard[np.arange(len(hard_prob)), hard_prob] = 1
        batch_hard = torch.tensor(batch_hard, dtype=torch.float32)  # (n, num_class)
        batch_hard = batch_hard.to(device)
        return batch_hard

# KL2: cosine
    def forward_and_infer_y_L2(self, X):
        mu = self.projectionHead(X)
        sigma = F.softplus(self.sigma_head(X))
        # reparameterization
        eps = torch.randn_like(mu)
        Z = mu + sigma * eps
        batch_y, logits_original, soft_prob = self.update_coarse_prototype(Z, get_soft_prob=True,softmax=False)
        batch_y = torch.tensor(batch_y, dtype=torch.float32).to(X.device)  # (n, num_class)
        KL2, loss_yz,  _, KL1 = self.KL_contrast(Z, batch_y,mu, sigma)
        return mu,soft_prob, Z, KL1, KL2, loss_yz, batch_y # loss_qy

        # return mu, soft_prob, Z, KL1, KL2, loss_coarse, batch_y  # loss_qy

    def forward_and_infer_conditionZ(self,X):
        mu = self.projectionHead(X)
        # mu = F.normalize(mu, dim=1, p=2)
        sigma = F.softplus(self.sigma_head(X))
        # sigma = self.sigma_head(X)
        # reparameterization
        eps = torch.randn_like(mu)
        Z = mu + sigma * eps
        KL1,KL2,KL3,batch_y,prob_y,prob_s = self.KL_contrast_conditioned_on_Z(Z,mu,sigma)
        return mu, prob_s, Z, KL1, KL2, KL3, batch_y  # loss_qy

    def KL1(self,mu,sigma,mu_s):
        # mu = F.normalize(mu,dim=1,p=2)
        # mu_s = F.normalize(mu_s,dim=1,p=2)
        sigma_s2 = self.proto_tau
        kl1 = torch.sum(torch.log(sigma**2/sigma_s2+1e-8)-sigma**2/sigma_s2,dim=1)+self.encode_length
        kl2 = torch.sum(((mu - mu_s) ** 2) / (sigma_s2), dim=1)

        kl = torch.sum(-0.5*(kl1-kl2),dim=-1)
        return kl

    def project2Q(self,X):
        Q = self.projectionHead(X)
        # Q = F.normalize(Q,dim=1,p =2)
        return Q

    def project2Z(self,X):
        mu = self.projectionHead(X)
        # mu = F.normalize(mu, dim=1, p=2)
        sigma = F.softplus(self.sigma_head(X))
        # reparameterization
        eps = torch.randn_like(mu)
        Z = mu + sigma * eps
        return Z

    def updata_coarse_prototype_direct(self,Z,softmax = False):
        coarse_prototype = self.mu_y
        logits_original = self.defined_sim(Z,coarse_prototype)
        if softmax:
            soft_prob = F.softmax(logits_original / self.consis_temp, dim=1)
        else:
            if self.sample_method == "softmax":
                soft_prob = F.softmax(logits_original / self.gumbel_temp, dim=1)


            elif self.sample_method == "gumbel_softmax":
                logits = F.softmax(logits_original, dim=1)
                logits = torch.log(logits + 1e-9)
                soft_prob = self.gumbel_softmax(logits, self.gumbel_temp, False)

        return soft_prob

    def update_coarse_prototype(self,Z,get_soft_prob = False,softmax = False):
        fine_prototype = self.mu_s
        logits_original = self.defined_sim(Z, fine_prototype)
        # ---cosine calculate prototypes
        if softmax:
            # soft_prob = F.softmax(logits_original / self.consis_temp, dim=1)
            logits = F.softmax(logits_original, dim=1)
            logits = torch.log(logits + 1e-9)
            soft_prob = self.gumbel_softmax(logits, self.consis_temp, False)
        else:
            if self.sample_method == "softmax":
                soft_prob = F.softmax(logits_original / self.gumbel_temp, dim=1)
            elif self.sample_method == "gumbel_softmax":
                logits = F.softmax(logits_original, dim=1)
                logits = torch.log(logits + 1e-9)
                soft_prob = self.gumbel_softmax(logits, self.gumbel_temp, False)

        hard_prob = self.get_hard_prob_in_all_Fine_books(soft_prob)

        fine_index = torch.nonzero(hard_prob)[:,1].detach().cpu().numpy()
        num = 0
        new_coarse_index = numpy.zeros(hard_prob.shape[0])
        for i in range(self.Fine_books.shape[0]):
            fine_num = int(self.Fine_books[i])
            coarse_index = (fine_index >= num) & (fine_index < num+fine_num)
            new_coarse_index[coarse_index] = i
            num += fine_num
        new_coarse_index = new_coarse_index.astype(np.int64)
        batch_y = np.zeros([Z.shape[0], self.Fine_books.shape[0]])
        batch_y[np.arange(Z.shape[0]), new_coarse_index] = 1
        if get_soft_prob:
            return batch_y, logits_original, soft_prob
        else:
            return batch_y,logits_original,hard_prob

    def update_coarse_prototype_conditioned_z(self, Z, get_logits=False,softmax=False,temp=None):
        fine_prototype = self.mu_s
        logits_original = self.defined_sim(Z, fine_prototype)
        if softmax:
            if temp is None:
                soft_prob = F.softmax(logits_original / self.consis_temp, dim=1)
            else:
                soft_prob = F.softmax(logits_original / temp, dim=1)
        else:
            if self.sample_method == "softmax":
                soft_prob = F.softmax(logits_original / self.gumbel_temp, dim=1)
            elif self.sample_method == "gumbel_softmax":
                logits = F.softmax(logits_original, dim=1)
                logits = torch.log(logits + 1e-9)
                soft_prob = self.gumbel_softmax(logits, self.gumbel_temp, False)

        hard_prob = self.get_hard_prob_in_all_Fine_books(soft_prob)
        logits_z_coarse = self.defined_sim(Z, self.mu_y)
        logits_s_coarse = self.defined_sim(hard_prob@fine_prototype,self.mu_y)
        logits_coarse = (logits_z_coarse + logits_s_coarse)/2
        if softmax:
            if temp is None:
                soft_prob_y = F.softmax(logits_coarse / self.gumbel_temp, dim=1)
            else:
                soft_prob_y = F.softmax(logits_coarse / temp, dim=1)
        else:
            if self.sample_method == "softmax":
                soft_prob_y = F.softmax(logits_coarse / self.consis_temp, dim=1) # (N,c)
            elif self.sample_method == "gumbel_softmax":
                logits_coarse = torch.log(F.softmax(logits_coarse, dim=1)+1e-9)
                # logits = torch.log(logits + 1e-9)
                soft_prob_y = self.gumbel_softmax(logits_coarse, self.gumbel_temp, False)
        batch_y = torch.argmax(soft_prob_y,dim=1)
        if get_logits:
            return batch_y, soft_prob_y, soft_prob,hard_prob,logits_s_coarse,logits_original
        else:
            return batch_y, soft_prob_y, soft_prob,hard_prob

    def KL_contrast_conditioned_on_Z(self,Z,mu,sigma):
        batch_y,prob_y,prob_s,hard_prob,logits_s_coarse,logits_original = self.update_coarse_prototype_conditioned_z(Z,softmax=False,get_logits=True)
        KL1 = 0
        KL2 = 0
        KL3 = 0
        device = batch_y.device
        #KL1
        KL1 += self.KL1(mu, sigma, hard_prob @ self.mu_s)

        mu_y = F.normalize(self.mu_y)
        mu_s = F.normalize(self.mu_s)
        # KL2
        #Euc
        # ls = -torch.log((hard_prob * F.softmax(logits_original / self.proto_tau, dim=1)).sum(dim=1))
        #Cosine
        logit2 = ((hard_prob @ mu_s) * Z).sum(1).view(-1, 1)  # (n, d)-sum->(n,1)
        ls_num = torch.exp(logit2 / self.proto_tau)  # (n, 1)
        # ls2_den = torch.exp(fine_prototype @ (q_fine.t()) / self.proto_tau).sum(0) # (c,n)
        ls2_den = (torch.exp((Z @ mu_s.t()) / self.proto_tau)).sum(1)
        # ls = - torch.log(ls_num / ls2_den.view(-1, 1))  # * q  # (n, k)
        ls = - torch.log(ls_num / ls2_den.view(-1, 1))
        ls = ls.sum()
        KL2 += ls
        # KL3
        # print(logit_c.shape)
        batch_y_index = np.zeros([Z.shape[0], self.Fine_books.shape[0]])
        batch_y_index[np.arange(Z.shape[0]), batch_y.detach().cpu().numpy()] = 1
        batch_y_index = torch.tensor(batch_y_index,dtype=torch.float32).to(device)
        # cosine
        logit_c = ((hard_prob @ mu_s) @ mu_y.t())  # [(N,d)@(d,c)]=(N,c)
        logit_c_den = torch.exp(logit_c / self.yz_tau)  #(N,c)
        logit_c_num = (batch_y_index * logit_c_den).sum(1)  # (N,1)*(N,c)=(N,c)-->(N,1)
       
        ls_coarse = -(torch.log(logit_c_num / logit_c_den.sum(1).view(-1, 1)))  
        ls_coarse = ls_coarse.sum()
        KL3 += ls_coarse
            # print(loss.a())
        KL1 = KL1/Z.shape[0]
        KL3 = KL3 / Z.shape[0]
        KL2 =KL2/Z.shape[0]
        return KL1,KL2,KL3,batch_y_index,prob_y,prob_s

    def KL_contrast(self,Z,batch_y,mu,sigma):
        sum = 0
        KL1 = 0
        KL2 = 0
        loss_coarse = 0
        prob_list = []
        # print(batch_y)
        device = batch_y.device
        batch_y = batch_y.detach().cpu().numpy()
        batch_y = batch_y.argmax(1)
        mu_y = F.normalize(self.mu_y)
        mu_s = F.normalize(self.mu_s)
        for i in range(self.Fine_books.shape[0]):
        # for i in range(len(self.Fine_books)):
            class_index = torch.tensor(np.argwhere(batch_y == i)).to(device).squeeze()
            Z_fine = Z[class_index].reshape(-1,self.encode_length)
            mu_fine = mu[class_index].reshape(-1,self.encode_length)
            sigma_fine = sigma[class_index].reshape(-1,self.encode_length)
            if Z_fine.shape[0] is 0:
                # print(i)
                fine_num = int(self.Fine_books[i])
                sum += fine_num
                continue
            # print(q_fine.shape)
            fine_num = int(self.Fine_books[i])
            fine_prototype = self.mu_s[sum:fine_num+sum] 
            coarse_y = np.zeros([1, self.Fine_books.shape[0]])
            coarse_y[:, i] = 1
            coarse_y = torch.tensor(coarse_y, dtype=torch.float32)  # (n, num_class)
            coarse_y = coarse_y.to(device)
            coarse_prototype = F.normalize(coarse_y@self.mu_y,p=2,dim=1)
            logits = self.defined_sim(Z_fine, fine_prototype)
           
            sum += fine_num
            if self.sample_method == "softmax":
                soft_prob = F.softmax(logits / self.gumbel_temp, dim=1)
                prob_list.append(soft_prob)

            elif self.sample_method == "gumbel_softmax":
                logits = F.softmax(logits, dim=1)
                logits = torch.log(logits + 1e-9)
                soft_prob = self.gumbel_softmax(logits, self.gumbel_temp, False)
                prob_list.append(soft_prob)
            hard_prob = self.get_hard_prob(soft_prob, i)
            n = Z_fine.shape[0]
            KL1 += self.KL1(mu_fine, sigma_fine, hard_prob @ fine_prototype)
            fine_prototype = F.normalize(fine_prototype, p=2, dim=1)
            Z_fine = F.normalize(Z_fine, p=2, dim=1)
            logit_c = (fine_prototype @ coarse_prototype.t())  # fine_num*1
        
            logit_c_num = torch.exp(logit_c / self.yz_tau) #(fine_num*1)
            logit_c_den = torch.exp((fine_prototype @ mu_y.t()) / self.yz_tau)  # fine_num*coarse_num
            ls_coarse =-soft_prob @(torch.log(logit_c_num / logit_c_den.sum(1).view(-1, 1))) 
            ls_coarse = ls_coarse.sum()/n
            loss_coarse += ls_coarse
            logit2 = ((hard_prob @ fine_prototype) * Z_fine).sum(1).view(-1, 1)  # (n, d)-sum->(n,1)
            ls_num = torch.exp(logit2 / self.proto_tau)  # (n, 1)
            ls2_den = (torch.exp((Z_fine @ mu_s.t()) / self.proto_tau)).sum(1)
            ls = - torch.log(ls_num / ls2_den.view(-1, 1))
            ls = ls.sum()
            KL2 += ls
        KL1 = KL1/Z.shape[0]
        loss_coarse = loss_coarse / Z.shape[0]
        KL2 =KL2/Z.shape[0]
        return KL2,loss_coarse,prob_list,KL1

class unsupHash_bert(unsupBaseModel_bert):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def define_parameters(self):
        self.pro_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, int(self.hparams.encode_length)),
            # nn.ReLU()
        )  # encoder
        self.decoder = nn.Sequential(
            nn.Linear(int(self.hparams.encode_length), self.hparams.hiddim),
            nn.ReLU(),
            nn.Linear(self.hparams.hiddim, 768)
        )
        self.vae_head = VAE_head( encode_length=self.hparams.encode_length, coarse_num=self.hparams.n_class, gumbel_temp=self.hparams.gumbel_temperature,
                               dist_metric=self.hparams.dist_metric, sample_method=self.hparams.sample_method,fine_Nums=self.hparams.fine_cluster_nums,
                               proto_tau=self.hparams.proto_tau,yz_tau=self.hparams.yz_tau,consis_temp=self.hparams.consis_temperature)
        self.criterion = NtXentLoss(self.hparams.tau)
        self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')
        self.angular = NtXentLoss_angular(self.hparams.tau,coarse_num=self.hparams.n_class)
        self.l2 = nn.MSELoss() # reconstruction loss

    def compute_prob_loss(self, prob0, prob1):
        def get_entropy(probs):
            q_ent = -(probs.mean(0) * torch.log(probs.mean(0) + 1e-12)).sum()
            return q_ent

        def get_cond_entropy(probs):
            q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(1).mean()
            return q_cond_ent
        q_ent0 = get_entropy(prob0)
        q_cond_ent0 = get_cond_entropy(prob0)
        q_ent1 = get_entropy(prob1)
        q_cond_ent1 = get_cond_entropy(prob1)
        wim_0 = q_ent0 - self.hparams.cond_ent_weight * q_cond_ent0
        wim_1 = q_ent1 - self.hparams.cond_ent_weight * q_cond_ent1
        wim = (wim_0 + wim_1) / 2.
        prob_loss = - wim
        return prob_loss

    def forward_with_no_augmentation(self,bert_0,labels_coarse,device):
        # bert_0 = self.get_embeddings(inputs, pooling=self.hparams.pooler_type)
        embd_0 = self.pro_layer(bert_0)
        labels_coarse = labels_coarse.detach().cpu().numpy()
        labels_coarse = labels_coarse.astype(np.int64)
        batch_y = np.zeros([len(labels_coarse), self.hparams.n_class])
        batch_y[np.arange(len(labels_coarse)), labels_coarse] = 1
        batch_y = torch.tensor(batch_y, dtype=torch.float32)  # (n, num_class)
        if self.hparams.cuda:
            # inputs = squeeze_dim(move_to_device(inputs, device), dim=1)
            batch_y = batch_y.to(device)
        mu_0,s_prob0, Z_0, KL11, KL12, loss_ys1, batch_y = self.vae_head.forward_and_infer_y_L2(embd_0)
        reconstruction_loss = self.reconstruction_loss(Z_0, bert_0)
        KL2 = KL12
        # use Q to compute angular normalization
        KL1 = KL11
        loss_yz = loss_ys1
        loss =  self.hparams.VAE_weight * (reconstruction_loss + self.hparams.KL_weight * (KL1 + KL2))

        return {'loss': loss, 'KL1(MSE)': KL1, 'KL2(Contrast)': KL2, 'YZ_Contrast': loss_yz,
                'reconstruct': reconstruction_loss}

    def consist_loss(self,prob_0,prob_1):
        sim = torch.sum(prob_0*prob_1,dim = 1) # dot product (N,1)
        loss = torch.log(sim + 1e-9)
        return -torch.mean( loss, dim=0)


    def reconstruction_loss(self,Z,targets):
        scores = self.decoder(Z)
        loss = ((scores-targets)**2).sum(dim=1).mean()
        return loss

    def forward_and_infer(self,bert_0,bert_1,device):
        embd_0 = self.pro_layer(bert_0)
        embd_1 = self.pro_layer(bert_1)  # z1,...zM
        mu_0, s_prob0, Z_0, KL11, KL12, KL13, batch_y = self.vae_head.forward_and_infer_conditionZ(embd_0)
        mu_1,s_prob1, Z_1, KL21, KL22, loss_ys2, batch_y_2 = self.vae_head.forward_and_infer_conditionZ(embd_1)
        reconstruction_loss = self.reconstruction_loss(Z_0, bert_0)
        KL2 = KL12
        KL1 = KL11
        KL3 = KL13
        L_consis = self.consist_loss(s_prob0,s_prob1)
        # # # L_mi = 0
        L_mi = self.compute_prob_loss(s_prob0,s_prob1)
        high_confi = torch.where(batch_y.argmax(1) == batch_y_2.argmax(1))[0]
        S_loss = L_consis + L_mi
        mu_y = self.vae_head.mu_y.to(device)
        coarse_selected = batch_y @ mu_y  # (batch_size, encode_length)
        coarse_selected = F.normalize(coarse_selected, p=2, dim=1)
        # use Q to compute angular normalization
        a_0 = F.normalize(F.normalize(mu_0, p=2, dim=1) - coarse_selected, p=2, dim=1)
        a_1 = F.normalize(F.normalize(mu_1, p=2, dim=1) - coarse_selected, p=2, dim=1)
        labels_coarse = batch_y.argmax(1)
        labels_coarse = labels_coarse.detach().cpu().numpy()
        labels_coarse = labels_coarse.astype(np.int64)
        if self.hparams.multi_queue:
            # contra_loss = self.angular(Q_0,Q_1,labels_coarse,device)
            # contra_loss = self.criterion(Q_0, Q_1)  # contrastive loss
            diff_loss = self.angular(mu_0, mu_1, labels_coarse, device, False)
            angular_loss = self.angular(a_0, a_1,labels_coarse, device)
            contra_loss = diff_loss + angular_loss
        else:
            contra_loss = self.criterion(mu_0, mu_1)  # contrastive loss
        n = high_confi.shape[0]
        self.high_confi_num += n
        loss = self.hparams.code_weight * contra_loss + self.hparams.VAE_weight * (
                reconstruction_loss  + self.hparams.KL_weight * (KL1 + KL2+ KL3)) + self.hparams.prob_weight * S_loss

        if self.hparams.multi_queue:
            return {'loss': loss, 'KL1': KL1, 'angular_loss': angular_loss, 'diff_loss': diff_loss,
                    'KL2(Contrast)': KL2, 'KL3': KL3, 'reconstruct': reconstruction_loss,
                    'CONSIS':L_consis,'MI':L_mi,'s_loss':S_loss}
        else:

            return {'loss': loss, 'KL1': KL1, 'diff_loss': contra_loss,
                    'KL2(Contrast)': KL2, 'kl3': KL3, 'reconstruct': reconstruction_loss,'s_loss':S_loss}


    def self_label_forward(self,bert_0,bert_1,device):
        embd_0 = self.pro_layer(bert_0)
        embd_1 = self.pro_layer(bert_1)  # z1,...zM
        Z_0 = self.vae_head.project2Q(embd_0)
        Z_1 = self.vae_head.project2Q(embd_1)
        batch_y, prob_y, prob_s, hard_prob = self.vae_head.update_coarse_prototype_conditioned_z(Z_0, softmax=True)
        batch_y1, prob_y1, prob_s1, hard_prob1 = self.vae_head.update_coarse_prototype_conditioned_z(Z_1, softmax=True,temp=5)
        high_confi_c = torch.where(prob_y.max(1)[0] > 0.9)[0]
        high_confi = torch.where(prob_s.max(1)[0] > 0.9)[0]
        # high_confi = torch.where((pred_1).max(1)[0]>0.9)[0]
        if high_confi.shape[0]!=0:
            fine_xe_loss = self.criterion_cls(prob_s1[high_confi], hard_prob[high_confi])
            # fine_xe_loss = 0
            # coarse_xe_loss = 0
        else:
            fine_xe_loss = 0
        if high_confi_c.shape[0]!=0:
            coarse_xe_loss = self.criterion_cls(prob_y1[high_confi_c], batch_y[high_confi_c])
            if math.isnan(coarse_xe_loss):
                print(coarse_xe_loss)
        else:
            coarse_xe_loss = 0
        L_mi = self.compute_prob_loss(prob_s1, prob_s)/2 +self.compute_prob_loss(prob_y1, prob_y)/2
        contra_loss = 0
        n = high_confi_c.shape[0]
        self.high_confi_num += n
        loss = coarse_xe_loss*0.5 +fine_xe_loss*0.5+ L_mi*1 + contra_loss
        # print({'loss': loss,'XE':coarse_xe_loss,'fine_xe':fine_xe_loss,'mi':L_mi,'contra':contra_loss})
        return {'loss': loss,'XE':coarse_xe_loss,'fine_xe':fine_xe_loss,'mi':L_mi,'contra':contra_loss}



    def encode_continuous(self, bert_inputs):
        embd = self.pro_layer(bert_inputs)
        return embd

    def encode_projection(self,bert_inputs):
        # bert_inputs = self.pro_layer2(bert_inputs)
        embd = self.pro_layer(bert_inputs)
        embd = self.vae_head.project2Q(embd)
        return embd

    def encode_Z_for_cluster(self,bert_inputs):
        embd = self.pro_layer(bert_inputs)
        embd = self.vae_head.project2Z(embd)
        return embd

    def encode_discrete(self, bert_inputs):
        # sigmoid
        embd = hash_layer(torch.sigmoid(self.encode_projection(bert_inputs)/self.hparams.hashing_alpha)-0.5)
        return embd


    def pq_fine_assignment(self,bert0,is_coarse):
        mu = self.encode_projection(bert0)
        if is_coarse:
            logits = self.vae_head.updata_coarse_prototype_direct(mu,softmax = True)
        else:
            _, _, logits = self.vae_head.update_coarse_prototype(mu,get_soft_prob=True,softmax=True)
        return logits


    def update_prorotypes_assignment(self):
        fine_original = self.vae_head.mu_s#.detach().cpu().numpy() # numpy:fine_num*dim
        coarse2fine_dict = {}
        coarse_prototypes = self.vae_head.mu_y
        device = coarse_prototypes.device
        for i in range(0, self.hparams.n_class):
            coarse2fine_dict.update({i: list()})
        # for i in range(fine_original.shape[0]):
        #     f_prototype = fine_original[i] #1*dim
        diff = fine_original.unsqueeze(1) - coarse_prototypes.unsqueeze(0)
        Euc_dist = - torch.sum(diff * diff, - 1) # f*c
        y_assignment = torch.argmax(Euc_dist,dim = 1) # f*1
        for i in range(0,fine_original.shape[0]):
            index = int(y_assignment[i])
            coarse2fine_dict.get(index).append(fine_original[i].detach().cpu().numpy())
        rearranged_fine_prototypes = []
        fine_nums = []
        for key in coarse2fine_dict.keys():
            rearranged_fine_prototypes.extend(coarse2fine_dict.get(key))
            fine_nums.append(len(coarse2fine_dict.get(key)))
        rearranged_fine_prototypes = np.array(rearranged_fine_prototypes)
        rearranged_fine_prototypes = torch.tensor(rearranged_fine_prototypes).to(device).squeeze()
        self.vae_head.clusrered_prototype_init(rearranged_fine_prototypes, coarse_prototypes, fine_nums)
        return None

    def init_prototypes_with_cluster_center(self,fine_prototypes,coarse_prototypes,fine_nums,requires=True):
        self.vae_head.clusrered_prototype_init(fine_prototypes,coarse_prototypes,fine_nums,requires=requires)

    def init_prototypes_with_original_cluster_center(self,requires=True):
        fine_books = self.vae_head.Fine_books.cpu()
        self.vae_head.clusrered_prototype_init(self.vae_head.mu_s,self.vae_head.mu_y,fine_books,requires=requires)

    def pred(self,bert_input):
        embd = self.encode_projection(bert_input)
        pred = self.predictor(embd)
        return pred


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        return optimizer

    def get_hparams_grid(self):
        grid = unsupBaseModel_bert.get_general_hparams_grid()
        grid.update({'gumbel_temperature': [5.0, 10.0],
                     'prob_weight': [0.1, 0.2, 0.3, 0.5]})
        # temperature,code_weight,cond_ent_weight
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = unsupBaseModel_bert.get_general_argparser()
        parser.add_argument("--hashing_alpha", default=1.0, type=float,
                            help="Temperature for sigmoid function [%(default)d]", )
        parser.add_argument("--gumbel_temperature", default=5.0, type=float,
                            help="gumbel_temperature [%(default)d]", )
        parser.add_argument("--consis_temperature", default=0.007, type=float,
                            help="consistancy_temperature [%(default)d]", )
        parser.add_argument("--dropout_rate", default=0.3, type=float,
                            help="Dropout rate [%(default)d]", )
        parser.add_argument("--pooler_type", default="cls", type=str)
        parser.add_argument("--dist_metric", default="euclidean", type=str)
        parser.add_argument("--code_weight", default=2.0, type=float)
        parser.add_argument("--ys_weight", default=1.0, type=float)
        parser.add_argument("--VAE_weight", default=1.0, type=float)
        parser.add_argument("--KL_weight", default=0.5, type=float)
        parser.add_argument("--cond_ent_weight", default=0.1, type=float)
        parser.add_argument("--prob_weight", default=0.1, type=float)
        parser.add_argument("--evaluate_metric", default="hamming", choices=["euclidean", "hamming"], type=str)
        parser.add_argument('--multi_queue',action="store_true")
        parser.add_argument("--sample_method", default="gumbel_softmax", type=str)
        parser.add_argument('--proto_tau', default=1.0, type=float, help="Temperature for protoNCE loss [%(default)d](sigma for s)")
        parser.add_argument('--yz_tau', default=1.0, type=float, help="Temperature for yz contrast(only needed when YZ_Contrast is true) [%(default)d]")
        parser.add_argument('--nclusters', default=26, type=int,help='the number of fine_class clusters')  # 3:yelp&arxiv;5:nyt



        return parser

class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return (torch.sign(input) + 1) // 2

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def  hash_layer(input):
    return hash.apply(input)