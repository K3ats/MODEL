##attack 
import torch
import numpy as np
import copy
from agg import tr_mean,crh
###fang attack
def attack_median_and_trimmedmean(benign_update,m):
    
    # benign_update = 
    agg_grads = torch.mean(benign_update, 0)
    deviation = torch.sign(agg_grads)
    device = benign_update.device
    b = 2
    max_vector = torch.max(benign_update, 0)[0]
    min_vector = torch.min(benign_update, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).to(device)
    min_ = (min_vector < 0).type(torch.FloatTensor).to(device)

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat(
        (max_vector[:, None], (max_vector * max_)[:, None]), dim=1
    )
    min_range = torch.cat(
        ((min_vector * min_)[:, None], min_vector[:, None]), dim=1
    )

    rand = (
        torch.from_numpy(
            np.random.uniform(0, 1, [len(deviation), m])
        )
        .type(torch.FloatTensor)
        .to(benign_update.device)
    )

    max_rand = (
        torch.stack([max_range[:, 0]] * rand.shape[1]).T
        + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    )
    min_rand = (
        torch.stack([min_range[:, 0]] * rand.shape[1]).T
        + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
    )

    mal_vec = (
        torch.stack(
            [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]
        ).T.to(device)
        * max_rand
        + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]
        ).T.to(device)
        * min_rand
    ).T
    return mal_vec
###
def multi_krum(all_updates, n_attackers, multi_k=False):

    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))

    aggregate = torch.mean(candidates, dim=0)

    return aggregate, np.array(candidate_indices)

def compute_lambda_fang(all_updates, model_re, n_attackers):

    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1)
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, :n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0])
    max_wre_dist = torch.max(torch.norm((all_updates - model_re), dim=1)) / (torch.sqrt(torch.Tensor([d]))[0])

    return (term_1 + max_wre_dist)

def get_malicious_updates_fang(all_updates, n_attackers):
    model_re=torch.mean(all_updates,0)
    deviation = torch.sign(model_re)
    lamda = compute_lambda_fang(all_updates, model_re, n_attackers)
    threshold = 1e-5

    mal_updates = []    
    while lamda > threshold:
        mal_update = (- lamda * deviation)

        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        _, krum_candidate = multi_krum(mal_updates, n_attackers)
        
        if krum_candidate < n_attackers:
            return mal_updates
        
        lamda *= 0.5

    if not len(mal_updates):
        print(lamda, threshold)
        mal_update = (model_re - lamda * deviation)
        
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates[:n_attackers]
###
from scipy.stats import norm
def LIE(para_list,m):
    p=copy.deepcopy(para_list[:m])
    n=para_list.shape[0]
    
    mu = torch.mean(para_list,0)
    sigma = torch.std(para_list,0)
    for i in range(m):
        p[i]=mu-norm.ppf((n/2-1)/(n-m))*sigma
    
    return p 
###    
def scaling_attack(para_list,m):
    p=copy.deepcopy(para_list[:m])
    factor= para_list.shape[1]
    for i in range(m):
        p[i]=para_list[i]*factor
    
    return p
### https://github.com/zaixizhang/FLDetector/blob/main/byzantine.py
def mean_attack(para_list,m):

    return torch.stack([-para_list[i] for i in range(m)])

def full_mean_attack(para_list,m):

    p=copy.deepcopy(para_list[:m])

    if m == para_list.shape[1]:
        return mean_attack(para_list,m)
    
    all_sum=torch.sum(para_list,0)
    m_para_sum= torch.sum(para_list[:m],0)
    for i in range(m):
        p[i]=((-all_sum- m_para_sum)/m)
    return p
#### attack for tailored
def AGR_tailored_attack_on_krum(all_updates, n_attackers, dev_type='unit_vec'):
    model_re = torch.mean(all_updates,0)
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([3.0])

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)
        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    return mal_updates

def AGR_tailored_attack_on_median(all_updates, n_attackers, dev_type='unit_vec'):
    model_re = torch.mean(all_updates,0)

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0])#compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = torch.median(mal_updates, 0)[0]
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    return mal_updates

def AGR_tailored_attack_on_trmean(all_updates, n_attackers, dev_type='unit_vec'):
    model_re = torch.mean(all_updates,0)
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]) #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0 
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = tr_mean(mal_updates, n_attackers)
        
        loss = torch.norm(agg_grads - model_re)
        
        if prev_loss < loss:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss
        
    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)

    return mal_updates
###attack for AGR unknow
def min_max(all_updates,m,dev_type='unit_vec'):
    model_re = torch.mean(all_updates,0)
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10]).float()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * m)
    return mal_updates

def min_sum(all_updates,m,dev_type='unit_vec'):
    model_re = torch.mean(all_updates,0)

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
    
    lamda = torch.Tensor([10.0]).float()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        score = torch.sum(distance)
        
        if score <= min_score:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print(lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * m)

    return mal_updates
###     
def fang_adap(all_para,n):
    l = 0.01
    max=torch.max(all_para,dim=0)[0]
    min=torch.min(all_para,dim=0)[0]
    x,y = all_para.shape
    m=np.zeros((n,y))
    for i in range(n):
        for y in range(y):
            m[i][y]=np.random.uniform(min[y],max[y])
    m=torch.from_numpy(m)
    para = torch.concat([all_para,m])
    v,_=crh(para)
    V = torch.zeros_like(para[0])
    for e in range(50):
        v_hat ,w = crh(para)
        # print(v_hat,w)
        m_weight = w[x:]
        for idx in range(n):
            for j in range(y):
                m[idx][j] += 2*l*(v_hat[j] - v[j])*m_weight[idx]/sum(w)
        para = torch.concat([all_para,m])
        if(abs(sum(v_hat-V))<1e-7 ):
            print("converge",e)
            V=copy.deepcopy(v_hat)
            break
        V=copy.deepcopy(v_hat)
    return m
