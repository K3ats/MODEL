#init
import torch
from model import *
from func import *
from train import *
from agg import *
from attack import *
from data_partion import Dataset_config
from torchvision.models import resnet18,ResNet18_Weights
import os
import pandas as pd
from sklearn.cluster import KMeans
#args
import argparse
def get_args():    
    parser = argparse.ArgumentParser()
    #important
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset,cifar,mnist")
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--num_users', type=int, default=50, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--pattern', type=str,  default='noniid', help='iid,noniid,1-9,iid-q')
    parser.add_argument('--dirichlet',type=float,default=0.5,help='dirichlet of noniid')
    parser.add_argument('--m_users', type=int,  default=20, help='number of malicious clients')
    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--ratio', type=float, default=1,  help="portion of iid user: ")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument("--sample", type=int, default=500, help="number of samples for each node")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # parser.add_argument('--alg',type=str,default='avg',help='avg or fed')
    
    args = parser.parse_args()
    return args

    

if __name__ == '__main__':
    args = get_args()
    # print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #dir 
    if args.pattern =='iid':
        dir = './result/{}_{}/{}'.format(args.dataset,args.model,args.pattern)
    else:
        dir = './result/{}_{}/{}_{}'.format(args.dataset,args.model,args.pattern,args.dirichlet)
    os.makedirs(dir,exist_ok=True)

    #data process
    dict_users, dataset_train, dataset_test= Dataset_config(args.dataset, args.num_users, args.pattern,args.dirichlet)
    np.save(dir+'/userdict.npy',dict_users)
    
    #model process
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob=resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features=net_glob.fc.in_features
        net_glob.fc=nn.Linear(num_features,10)
        model=net_glob.to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fashion':
        net_glob = CNNfashion().to(args.device)   
    # elif args.model == 'vgg' and args.dataset == 'cifar':
    #     net_glob = VGG11().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # writer = SummaryWriter(log_dir='./logs')
    # net_glob.requires_grad_()
    net_glob.train()

# copy weights
    w_glob = net_glob.state_dict()
    
#local train
    if os.path.exists(dir+'/50e_w_locals.npy'):
        print('file exist')
        w_locals = np.load(dir+'/50e_w_locals.npy',allow_pickle=True)[()]
    elif args.model == 'resnet' and args.dataset == 'cifar' :
        print('file exist')
        w_locals = np.load('/home/k3ats/jupyter/50c_resnet_cifar10_noniid0.5.npy',allow_pickle=True)[()]
    else:
        w_locals = []
        locals_acc=[]
        idxs_users = range(args.num_users)
        pre_para = copy.deepcopy(net_glob.to(args.device).state_dict())
        for idx in idxs_users:
            args.local_ep = 50
            net_glob.load_state_dict(pre_para)
            local = LocalUpdate(args=args, dataset=dataset_train, train=dict_users[idx])
            local_net, w, _= local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            #local val
            net_glob.load_state_dict(w)
            val_acc,val_loss = local.val(local_net,args)
            print(val_acc,val_loss)
            locals_acc.append(val_acc)
            
        np.save(dir+'/50e_w_locals.npy',w_locals)
        np.save(dir+'/local_acc.npy',locals_acc)

#exp
    ddir='./save/{}_{}_{}_{}'.format(args.dataset,args.model,args.pattern,args.m_users)
    os.makedirs(ddir,exist_ok=True)
    update =w_locals
    update_w = torch.stack([flatten(i).cpu() for i in update])
    m=args.m_users
    n_attacker = args.m_users
    writer = pd.ExcelWriter(ddir+'/result.xlsx')
    print('----------------table1----------------')
    table1_attack=[AGR_tailored_attack_on_krum,AGR_tailored_attack_on_trmean,
               attack_median_and_trimmedmean,get_malicious_updates_fang,LIE]
    sheet1=[]    
    t1_record = []
    for attack in table1_attack:            
        poison = attack(update_w,m)
        uw = torch.vstack([poison,update_w[m:]])
        record=[]
        c_record = []
        #mean
        net_glob.load_state_dict(unflatten(torch.mean(uw,dim=0),update[0]))
        # print(attack.__name__,'mean',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        # median
        net_glob.load_state_dict(unflatten(torch.median(uw,dim=0)[0],update[0]))
        # print(attack.__name__,'median',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        #trim
        final_w = tr_mean(uw,n_attacker)
        net_glob.load_state_dict(unflatten(final_w,update[0]))
        # print(attack.__name__,'trim',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        #krum
        if m ==24 :
            final_w = multi_krum_defence(uw,20)
        else:
            final_w = multi_krum_defence(uw,n_attacker)
        net_glob.load_state_dict(unflatten(final_w,update[0]))
        # print(attack.__name__,'krum',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        #bulyan
        # print('bulyan')
        # if(m==24):
        #     final_w = bulyan(uw,20)
        # else:
        #     final_w = bulyan(uw,n_attacker)
        # net_glob.load_state_dict(unflatten(final_w,update[0]))
        # # print(attack.__name__,'bulyan',test_img(net_glob,dataset_test,args))
        # record.append(test_img(net_glob,dataset_test,args)[0])
        
        #dnc
        benign_ids,final_w = dnc(uw,n_attacker)
        c_record.append(benign_ids)
        net_glob.load_state_dict(unflatten(final_w,update[0]))
        # print(attack.__name__,'dnc',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        
        #TD
        g,distance= crh(uw)
        
        #TDFL-cos
        # p=TDFL_cos(uw,0.95)
        cs=[]
        for idx in range(len(uw)):
            cs.append(torch.cosine_similarity(uw[idx],g,dim=0))
        cs = torch.stack(cs)
        print(cs)
        p= np.where(cs>=0.95)
        
        c_record.append(p)
        net_glob.load_state_dict(unflatten(torch.mean(uw[p],dim=0),update[0]))
        # print(attack.__name__,'TDFL_cos',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        #one_crh
        # p1=one_crh(uw)
        distance = distance.numpy()
        a=np.where(distance > np.mean(distance)-0.1)
        if len(a[0])>(uw.shape[0]/2):
            p1= a
        else : p1= np.delete(np.arange(len(uw)),a)
        
        c_record.append(p1)
        net_glob.load_state_dict(unflatten(torch.mean(uw[p1],dim=0),update[0]))
        # print(attack.__name__,'one_crh',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        #kmeanscrh
        # p2=kmeans_crh(uw)
        
        re = KMeans(2, random_state=0, n_init="auto").fit(distance.reshape(-1,1)).labels_
        print(re)
        if len(np.where(re==0)[0])>len(np.where(re==1)[0]):
            p2= np.where(re==0)[0]
        else:
            p2= np.where(re==1)[0]
            
        c_record.append(p2)
        net_glob.load_state_dict(unflatten(torch.mean(uw[p2],dim=0),update[0]))
        # print(attack.__name__,'kmeans_crh',test_img(net_glob,dataset_test,args))
        record.append(test_img(net_glob,dataset_test,args)[0])
        
        sheet1.append(copy.deepcopy(record))
        t1_record.append(copy.deepcopy(c_record))
    np.save(ddir+'/table1.npy',sheet1)
    np.save(ddir+'/t1_record.npy',t1_record)
    #save to excel
    f = pd.DataFrame(sheet1,
                    index=['AGR_tailored_attack_on_krum','AGR_tailored_attack_on_trmean','attack_median_and_trimmedmean',
                        'get_malicious_updates_fang','LIE']
                 ,columns=['mean','median','trim-mean','krum','dnc','TDFL_cos','one_TD','K-TD']
                    ).astype(np.float64)
    print(f)
    f.to_excel(writer,sheet_name="table1")

    writer.close()

