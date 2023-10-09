#update
import copy
import math

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from sklearn import metrics


# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, train=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.len_train = len(train)
        self.len_val = int(self.len_train*0.2)
        self.ldr_train = DataLoader(dataset,batch_size=self.args.local_bs, drop_last=True,sampler=torch.utils.data.SubsetRandomSampler(train))
        self.ldr_val = DataLoader(dataset,batch_size=self.args.local_bs, drop_last=True,sampler=torch.utils.data.SubsetRandomSampler(train[:self.len_val]))

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()  # 将模型的参数梯度初始化为0
                log_probs = net(images)  # MLP.forward(net,images)向前传播计算预测值
                loss = self.loss_func(log_probs, labels)  # 计算损失
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新参数
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), self.len_train,
                              100. * batch_idx / self.len_train, loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net, net.state_dict(), sum(epoch_loss) / len(epoch_loss) # 返回参数和loss,
    
    def val(self,net, args):
        net.eval()
        # testing
        test_loss = 0
        correct = 0
        # data_loader = DataLoader(datatest, batch_size=args.bs)
        for idx, (data, target) in enumerate(self.ldr_val):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= self.len_val
        accuracy = correct / self.len_val

        return accuracy, test_loss
