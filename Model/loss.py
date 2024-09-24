import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NtXentLoss(nn.Module):
    def __init__(self, temperature):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).cuda().long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class NtXentLoss_angular(nn.Module):
    def __init__(self, temperature, coarse_num):
        super(NtXentLoss_angular, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.coarse_num =coarse_num


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def mark_samples_discarded(self, batch_size, labels_coarse,keepSame=True):
        if keepSame:
            return self.mark_samples_discarded_sameClass(batch_size,labels_coarse)
        else:
            return self.mark_samples_discarded_diffClass(batch_size,labels_coarse)


    def mark_samples_discarded_diffClass(self, batch_size,labels_coarse):
        # batch_y: batch_size*1
        # labels_coarse = labels_coarse.detach().cpu().numpy()
        # labels_coarse = labels_coarse.astype(np.int64)
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        # print(labels_coarse)
        mask = mask.fill_diagonal_(0)
        for i in range(self.coarse_num):
            class_index = np.argwhere(labels_coarse==i) # sample_num_of_coarse_class_i *1 ,each position is the corresponding index
            # print(i)
            # print(class_index)
            for j in range(len(class_index)):
                index1 = int(class_index[j])
                # if len(class_index) == 1:
                #     print(index1)
                #
                #     print("--------------------")
                for t in range(j,len(class_index)):
                    index2 = int(class_index[t])
                    if index1 ==index2:
                        continue
                    mask[index1, index2] = 0
                    mask[index2, index1] = 0
                    mask[index1,index2+batch_size]= 0
                    mask[index2,index1+batch_size]= 0

                    mask[index1+batch_size,index2] =0
                    mask[index1 +batch_size, index2+batch_size] = 0
                    mask[index2+batch_size,index1]=0
                    mask[index2+batch_size,index2+batch_size]=0

        return mask

    def mark_samples_discarded_sameClass(self, batch_size,labels_coarse):

        N = 2 * batch_size
        mask = torch.zeros((N, N), dtype=bool)

        for i in range(self.coarse_num):
            class_index = np.argwhere(labels_coarse==i) # sample_num_of_coarse_class_i *1 ,each position is the corresponding index
            for j in range(len(class_index)):
                index1 = int(class_index[j])
                # mask[index1, index1] = 0
                mask[index1, index1+batch_size] = 1
                mask[index1 + batch_size,index1] = 1
                # if len(class_index) == 1:
                #     print(index1)
                #
                #     print("--------------------")
                for t in range(j,len(class_index)):
                    index2 = int(class_index[t])
                    if index1!=index2:
                        mask[index1, index2] = 1
                        mask[index2, index1] = 1
                    mask[index1,index2+batch_size]= 1
                    mask[index2,index1+batch_size]= 1

                    mask[index1+batch_size,index2] =1
                    mask[index1 +batch_size, index2+batch_size] = 1
                    mask[index2+batch_size,index1]=1
                    mask[index2+batch_size,index2+batch_size]=1

        return mask


# keepSame = ture: y same as negatives
    def forward(self, z_i, z_j,labels_coarse,device,keepSame=True):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        non_negative_mask = self.mark_samples_discarded(batch_size,labels_coarse,keepSame) #N*N
        non_negative_mask = non_negative_mask.to(device)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        neg_den = torch.exp(sim)*non_negative_mask #N*N
        neg_den = torch.sum(neg_den,dim=1) # N*1
        logits = positive_samples #N,1
        logits = torch.exp(logits) #N*(N-2) #N,1
        neg_den = neg_den.unsqueeze(1)
        loss = torch.div(logits,neg_den)
        loss = (-torch.log(loss)).sum()/N
        if math.isnan(loss):
            # print(neg_den)
            print(non_negative_mask)
            print(sim)
            print(z)
        return loss

class NtXentLoss_diff(nn.Module):
    def __init__(self, temperature, coarse_num):
        super(NtXentLoss_diff, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.coarse_num =coarse_num


    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask


    def mark_samples_discarded(self, batch_size,labels_coarse):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)

        for i in range(self.coarse_num):
            class_index = np.argwhere(labels_coarse==i) # sample_num_of_coarse_class_i *1 ,each position is the corresponding index
            # print(i)
            # print(class_index)
            for j in range(len(class_index)):
                index1 = int(class_index[j])
                for t in range(j,len(class_index)):
                    index2 = int(class_index[t])
                    if index1 !=index2:
                        mask[index1, index2] = 0
                        mask[index2, index1] = 0


                    mask[index1,index2+batch_size]= 0
                    mask[index2,index1+batch_size]= 0

                    mask[index1+batch_size,index2] =0
                    mask[index1 +batch_size, index2+batch_size] = 0
                    mask[index2+batch_size,index1]=0
                    mask[index2+batch_size,index2+batch_size]=0

        return mask



    def forward(self, z_i, z_j,labels_coarse,device):
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        non_negative_mask = self.mark_samples_discarded(batch_size,labels_coarse) #N*N
        non_negative_mask = non_negative_mask.to(device)
        # print(non_negative_mask.shape)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        neg_den = torch.exp(sim)*non_negative_mask #N*N
        # print(neg_den)
        neg_den = torch.sum(neg_den,dim=1) # N*1
        logits = positive_samples #N,1
        logits = torch.exp(logits) #N*(N-2) #N,1
        neg_den = neg_den.unsqueeze(1)
        # loss = logits/den
        loss = torch.div(logits,neg_den)
        loss = (-torch.log(loss)).sum()/N
        if math.isnan(loss):
            # print(neg_den)
            print(non_negative_mask)
            print(sim)
            print(z)
        return loss


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    # init = init / (init.norm(2, dim=1).unsqueeze(1) + 1e-8)
    return init

def init(shape,prototype):
    init_range = torch.max(torch.max(prototype),-torch.min(prototype))

def list_glorot(shape,coarse_prototype):
    #list_glorot([fine_Nums,N_books*L_word],self.mu_y)
    # fine_prototype_Nums_list * dim
    # e.g. [2,4,5,1]*128
    prototype = []
    print(shape[0])
    print(len(shape))
    for i in range(len(shape[0])):
        prototype_nums = shape[0][i]
        init_range = torch.nn.init.normal_(torch.Tensor(prototype_nums,shape[1]))
        init = coarse_prototype[i]+init_range
        init = init.detach().cpu().numpy()
        # init_range = np.sqrt(6.0 / (prototype_nums+ shape[1]))
        # init = (2 * init_range) * torch.rand(prototype_nums, shape[1]) - init_range
        prototype.extend(init)
    prototype = np.array(prototype,dtype='float32')
    # prototype = torch.from_numpy(prototype)
    prototype = torch.Tensor(prototype)
    # prototype = prototype.to('cuda')
    print(prototype.shape)
    # print(len(prototype[0]))
    # print(len(prototype))
    return prototype


def anchor_glorot(shape,coarse_prototype,anchor_rep):
    #list_glorot([fine_Nums,N_books*L_word],self.mu_y)
    # fine_prototype_Nums_list * dim
    # e.g. [2,4,5,1]*128
    device = coarse_prototype.device
    prototype = []
    print(shape[0])
    print(len(shape))
    sum = 0
    for i in range(len(shape[0])):
        prototype_nums = shape[0][i]
        init_range = torch.nn.init.normal_(torch.Tensor(prototype_nums,shape[1])).to(device)
        anchor_init_range = anchor_rep[sum:sum+prototype_nums].to(device)
        sum+=prototype_nums
        init = coarse_prototype[i]+init_range+anchor_init_range
        init = init.detach().cpu().numpy()
        # init_range = np.sqrt(6.0 / (prototype_nums+ shape[1]))
        # init = (2 * init_range) * torch.rand(prototype_nums, shape[1]) - init_range
        prototype.extend(init)
    prototype = np.array(prototype,dtype='float32')
    # prototype = torch.from_numpy(prototype)
    prototype = torch.Tensor(prototype).to(device)
    # prototype = prototype.to('cuda')
    print(prototype.shape)
    # print(len(prototype[0]))
    # print(len(prototype))
    return prototype

def y_glorot(fine_prototypes,fine_Nums):
    #list_glorot([fine_Nums,N_books*L_word],self.mu_y)
    # fine_prototype_Nums_list * dim
    # fine_prototype [fine_Nums,N_books*L_word]
    # e.g. [2,4,5,1]*128
    # fine_Nums: [2,4,5,1]
    # print(fine_Nums)
    coarse_num = len(fine_Nums)
    prototype = []
    count_fine_num = 0
    for i in range(coarse_num):
        fine_num = fine_Nums[i] # the fine prototype num in a coarse class
        # print(fine_num)
        fine_prototype = fine_prototypes[count_fine_num:count_fine_num+fine_num]
        # print(fine_prototype)
        # print(fine_prototype.shape)
        coarse_prototype = torch.mean(fine_prototype,dim=0,keepdim=True).detach().cpu().numpy()
        # print(coarse_prototype.shape)
        prototype.extend(coarse_prototype)
        count_fine_num+=fine_num
    prototype = np.array(prototype, dtype='float32')
    # prototype = torch.from_numpy(prototype)
    prototype = torch.Tensor(prototype)
    # prototype = prototype.to('cuda')
    # print(prototype.shape)
    return prototype



class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


