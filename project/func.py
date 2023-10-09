#test
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np


def test_img(net_g, datatest, args,sampler=None):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if sampler is not None :
        data_loader = DataLoader(datatest, batch_size=args.bs,sampler=sampler)
    else:
        data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = correct.item() / 100

    return accuracy, test_loss

def test_dis(net,dataset,args):
    net.eval()
    data_loader = DataLoader(dataset,batch_size=args.bs)
    y=[]
    for data,target in data_loader:
        data, target = data.cuda(), target.cuda()
        predits = net(data)
        y.append(predits.data.max(1)[1].cpu().numpy())
    
        
    return Counter(np.concatenate(y))

def flatten(idx_dict):
    return torch.concat([torch.flatten(idx_dict[key]) for key in idx_dict])
# a = flatten(w_locals[0])
# print(a,a.shape)


def unflatten(flattened, normal_shape):
    w_local = {}
    for k in normal_shape:
        n = len(normal_shape[k].view(-1))
        w_local[k] = (flattened[:n].reshape(normal_shape[k].shape)).clone().detach()
        flattened=flattened[n:]
    return w_local