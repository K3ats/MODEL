#defences
import numpy as np
import torch
import copy

def multi_krum_defence(all_updates, n_attackers, multi_k=False):
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

    # return aggregate, np.array(candidate_indices)
    return aggregate

def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return out

def bulyan(all_updates, n_attackers):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0), np.array(candidate_indices)
    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)

def dnc(updates,n_attackers):
    d = len(updates[1])
    num_iters = 1
    sub_dim= 1000
    fliter_frac=1.0
    benign_ids = []
    for i in range(num_iters):
        indices = torch.randperm(d)[: sub_dim]
        sub_updates = updates[:, indices]
        mu = sub_updates.mean(dim=0)
        centered_update = sub_updates - mu
        v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
        s = np.array(
            [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
        )

        good = s.argsort()[
            : len(updates) - int(fliter_frac * n_attackers)
        ]
        benign_ids.extend(good)
        print(benign_ids)
    
    #
    benign_ids = list(set(benign_ids))
    benign_updates = updates[benign_ids, :].mean(dim=0)
    return benign_ids,benign_updates

#
def TDFL_cos(uw,t):
    g,_=crh(uw,None)
    cs=[]
    for idx in range(len(uw)):
        cs.append(torch.cosine_similarity(uw[idx],g,dim=0))
    cs = torch.stack(cs)
    print(cs)
    return np.where(cs>=t)
    
###our 
##coming soon

