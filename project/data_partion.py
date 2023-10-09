#sample
import numpy as np
from torchvision import datasets, transforms
import torch
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
def Dataset_config(dataset, num_users, pattern,d):
    if dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(root='./data/', train=True, download=True,
                                       transform=trans_mnist)
        dataset_test = datasets.MNIST(root='./data/', train=False, download=True,
                                      transform=trans_mnist)
        X_train, y_train = dataset_train.data, dataset_train.targets
        X_train = X_train.data.numpy()
        y_train = y_train.data.numpy()

    elif dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        dataset_train = datasets.CIFAR10('./data/', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('./data/', train=False, download=True, transform=transform_test)
                
        X_train, y_train = dataset_train.data, dataset_train.targets
        y_train = np.array(y_train)
        
    elif dataset =='fashion':
        transformations = transforms.Compose([transforms.ToTensor(),])
        dataset_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformations)
        dataset_test = datasets.FashionMNIST('./data', download=True, train=False, transform=transformations)
        X_train, y_train = dataset_train.data, dataset_train.targets
                     
        
    else:
        exit('Error: unrecognized dataset')

    if pattern == 'iid':
        dict_users = iid(y_train, num_users)

    elif pattern == 'noniid':
        dict_users = noniid(y_train, num_users,d)
    elif pattern > "1" and pattern <= "9":
        #todo labelnoniid
        # exit('Error: unfinsh')
        dict_users=label(pattern,num_users,y_train)
    elif pattern == "iid-q":
        idxs = np.random.permutation(y_train.shape[0])
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(0.5, num_users))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch= np.split(idxs,proportions)
        dict_users = {i:batch[i] for i in range(num_users)}
    else:
        exit('Error: unrecognized pattern')
        
    
    # all_idxs = [i for i in range(y_train.shape[0])]
    # l=len(len(dataset_test))
    # alp=0.8
    # all_idxs_test = [i for i in range(l)]
    # train_sampler = SubsetRandomSampler(all_idxs[:int(l*alp)])
    # val_sampler = SubsetRandomSampler(all_idxs[int(l*alp):l])
    # test_sampler = SubsetRandomSampler(all_idxs_test)

    # print(dict_users)
    #for val
    # train_dict = {}
    # val_dict={}
    # for i in range(num_users):
    #     t,v = random_split(dict_users[i],lengths=[0.8,0.2],generator=torch.Generator().manual_seed(42))
    #     train_dict[i] = t
    #     val_dict[i] = v
    return dict_users, dataset_train, dataset_test
    # , dataset_test_part


def iid(dataset, num_users):
    """
    iid partition
    """
    print("iid partion")
    idxs = np.random.permutation(dataset.shape[0])
    batch = np.array_split(idxs,num_users)
    dict_users = {i:batch[i] for i in range(num_users)}
    return dict_users


def noniid(dataset, num_users,d):
    """
    noniid partition
    dirichlet=0.5
    """
    min_size = 0
    min_require_size = 10
    K = 10
    # dirichlet = 0.9
    print("noniid partion",d)
    N = dataset.shape[0]
    #np.random.seed(2020)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(dataset== k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(d, num_users))
            
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    return net_dataidx_map

def label(partition:str,num_users,y_train):
    num = eval(partition)
    K = 10
    if num == 10:
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
        for i in range(10):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,num_users)
            for j in range(num_users):
                net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
    else:
        times=[0 for i in range(K)]
        contain=[]
        for i in range(num_users):
            current=[i%K]
            times[i%K]+=1
            j=1
            while (j<num):
                ind=np.random.randint(0,K-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_users)}
        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(num_users):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1
    return net_dataidx_map