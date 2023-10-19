# -*- coding: utf-8 -*-
# A quick test for demonstration (pre-trained model used)
# Free to migrate it to other FL scenarios

import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import *
from models.test import test_img

sys.path.insert(0,'utils/')

from logger import *
from eval import *
from misc import *

from cifar10_normal_train import *
from cifar10_util import *
from adam import Adam
from sgd import SGD
args = args_parser()
def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)
def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)
arch = 'alexnet'



# w1, w2 and w3 represent global models with different precision, respectively
net_glob, _ = return_model(arch, 0.1, 0.9, parallel=False)
net_glob.load_state_dict(torch.load('weight/w3.pkl'))

test_net, _ = return_model(arch, 0.1, 0.9, parallel=False)

target_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
target_model.load_state_dict(torch.load('weight/w1.pkl'))

#cifar10+alexnet
trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10('data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10('data/cifar', train=False, download=True, transform=trans_cifar)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(len(dataset_train)/10))
data_batch, _ = next(iter(train_loader))
data_batch=data_batch.cuda()
acc_train, loss_train = test_img(net_glob, dataset_train, args)
acc_test, loss_test = test_img(net_glob, dataset_test, args)
print("begin accuracy: {:.2f}".format(acc_train))
# print("Testing accuracy: {:.2f}".format(acc_test))

loss_train = []
cv_loss, cv_acc = [], []
val_loss_pre, counter = 0, 0
net_best = None
best_loss = None
val_acc_list, net_list = [], []

#Fedavg
agr='Fedavg'
c=2
loss_name='l2' #l2 umap
pre=[]
i=0
decay=1.1

def AGR(w_locals):
    if agr=='Fedavg':
        return FedAvg(w_locals)
    elif agr=='Median':
        return Median(w_locals)
    elif agr=='norm_bounding':
        return norm_bounding(w_locals, args.num_users)
    elif agr=='Trmean':
        return trimmed_mean(w_locals,args.num_users)
    elif agr=='bulyan':
        return bulyan(w_locals, args.num_users)
    elif agr=='krum':
        return multi_krum(w_locals, args.num_users, multi_k=False)
    elif agr=='Mkrum':
        return multi_krum(w_locals, args.num_users, multi_k=True)
    elif agr=='Fltrust':
        return FLtrust(w_locals,args.num_users,w_locals[0])
    elif agr=='CC':
        return CC(w_locals,net_glob,aggregator)
    elif agr=='DNC':
        return DNC(w_locals,net_glob)
    elif agr=='Signguard':
        return Signguard(w_locals,net_glob)




def umap(output, target, data_batch, eps=0.0000001):
    start_idx = 0
    for param in test_net.parameters():
        length = len(param.data.view(-1))
        param.data = output[start_idx: start_idx + length].reshape(param.data.shape).cuda()
        start_idx = start_idx + length

    output_net, _ = test_net(data_batch)
    target_net, _ = target(data_batch)
    # Normalize each vector by its norm
    (n, d) = output_net.shape
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    # model_similarity = model_similarity - torch.min(model_similarity,dim=1)[0].view(-1,1)
    model_distance = 1-model_similarity #[0,2]
    model_distance[range(n), range(n)] = 3
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0

    model_similarity = 1-model_distance

    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1-target_similarity
    target_distance[range(n), range(n)] = 3
    target_distance = target_distance - torch.min(target_distance,dim=1)[0].view(-1,1)
    target_distance[range(n), range(n)] = 0
    target_similarity = 1 - target_distance


    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = CE(target_similarity,model_similarity)
    # print(target_similarity,model_similarity)
    # exit()
    return loss



def test(w_glob):
    start_idx = 0
    for param in test_net.parameters():
        length = len(param.data.view(-1))
        param.data = w_glob[start_idx: start_idx + length].reshape(param.data.shape).cuda()
        start_idx = start_idx + length
    acc_train, loss_train = test_img(test_net, dataset_train, args)
    print("test accuracy: {:.2f}".format(acc_train))


def calculate_distance(w_glob,target):
    distance=torch.norm(w_glob - target, p=2)
    # print("now distance: "+str(distance.item()))
    return distance.item()

def calculate_malicious(target_model,nusers,w_locals,choice,p):
    min_dis = 1000000
    fed_model_para = []
    target_model_para = []
    global pre
    global i
    for param in net_glob.parameters():
            fed_model_para = param.data.view(-1) if not len(fed_model_para) else torch.cat(
                (fed_model_para, param.data.view(-1)))
    for param in target_model.parameters():
            target_model_para = param.data.view(-1) if not len(target_model_para) else torch.cat(
                (target_model_para, param.data.view(-1)))

    while True:
        w_locals_c = w_locals
        # Three attack primitives
        if choice == 1:
            w0 = p * torch.randn_like(target_model_para)
        elif choice == 2:
            w0 = p * target_model_para
        elif choice == 3:
            # Note that here is a simulation in the full-knowledge setting
            w0 = p * (target_model_para - torch.sum(w_locals, dim=0) / nusers)

        # print("current p:"+str(p))
        while len(w_locals_c) < nusers:
            w_locals_c = w0[None, :] if len(w_locals_c) == 0 else torch.cat((w_locals_c, w0[None, :]), 0)
        w_glob = AGR(w_locals_c)
        # test(w_glob)
        if loss_name=='umap':
            now_dis = umap(w_glob, target_model, data_batch)
        elif loss_name=='l2':
            now_dis = calculate_distance(w_glob, target_model_para)
        # print(now_dis)
        # print('step 1:'+str(now_dis))
        start_idx=0
        for param in test_net.parameters():
            length=len(param.data.view(-1))
            param.data = w_glob[start_idx: start_idx+length].reshape(param.data.shape).cuda()
            start_idx = start_idx + length
        if now_dis<=min_dis:
            min_dis=now_dis
            p/=decay
        else :
            p*=decay
            break

    malicious_model= w0

    return malicious_model


if __name__ == '__main__':
    aggregator = Clipping(tau=5, n_iter=1)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = cifar_iid(dataset_train, args.num_users)
    temperature=1
    for iter in range(args.epochs):
        loss_locals = []
        w_locals = []
        w_benign_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users=np.arange(0, args.num_users)
        for idx in idxs_users:
            w0 = []
            if idx==args.num_users*4//5:
                w0=calculate_malicious(target_model,args.num_users,w_locals,c,10)
                print("choice:"+str(c))
                print("loss:"+str(loss_name))
                break
            else :
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                for param in w.parameters():
                    w0 = param.data.view(-1) if not len(w0) else torch.cat(
                        (w0, param.data.view(-1)))

            w_locals = w0[None, :] if len(w_locals) == 0 else torch.cat(
                    (w_locals, w0[None, :]), 0)


        while len(w_locals)<args.num_users:
            w_locals = w0[None, :] if len(w_locals) == 0 else torch.cat((w_locals, w0[None, :]), 0)

        w_glob = AGR(w_locals)

        start_idx=0
        for param in net_glob.parameters():
            length=len(param.data.view(-1))
            param.data = w_glob[start_idx: start_idx+length].reshape(param.data.shape).cuda()
            start_idx = start_idx + length


        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print(str(iter)+"  end accuracy: {:.2f}".format(acc_train))

