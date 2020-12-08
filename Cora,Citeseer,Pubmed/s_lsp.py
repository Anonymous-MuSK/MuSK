from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model_plus import *
import uuid
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
parser.add_argument('--lbd_lsp', type=float, default=0)
parser.add_argument('--kernel', default='kl', help='kl,lin,poly,dist,RBF')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

nlayers=8
lbd_LSP = args.lbd_lsp
# Load data
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
t_PATH = "./teacher/teacher_"+str(args.data)+str(args.layer)+".pth"
checkpt_file = "./student/studentLSP_"+str(args.data)+str(args.layer)+str(lbd_LSP)+str(nlayers)+".pth"
#print(cudaid,checkpt_file)


teacher = GCNII(nfeat=features.shape[1],
                        nlayers=args.layer,
                        nhidden=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        lamda = args.lamda,
                        alpha=args.alpha,
                        variant=args.variant).to(device)
teacher.load_state_dict(torch.load(t_PATH))
model = GCNII(nfeat=features.shape[1],
                nlayers=nlayers,
                nhidden=args.hidden,
                nclass=int(labels.max()) + 1,
                dropout=args.dropout,
                lamda = args.lamda,
                alpha=args.alpha,
                variant=args.variant).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)


def count_params(student):
    return sum(p.numel() for p in student.parameters() if p.requires_grad)

kl_op_batch = torch.nn.KLDivLoss(reduction='batchmean')
temperature = 2

def train():
    teacher.eval()
    model.train()
    optimizer.zero_grad()

    # loss_CE
    t_output, t_hidden = teacher(features,adj)
    s_output, s_hidden = model(features,adj)
    loss_CE = F.nll_loss(s_output[idx_train], labels[idx_train].to(device))
    acc_train = accuracy(s_output[idx_train], labels[idx_train].to(device))

    # loss_lsp
    N = t_hidden[idx_train].size()[0]
    t_dist = torch.zeros(t_hidden.size()[0], t_hidden.size()[0])
    s_dist = torch.zeros(s_hidden.size()[0], s_hidden.size()[0])

    LS = list()
    train_adj = adj.to_dense()
    train_adj = train_adj[idx_train][:, idx_train]
    row, col = dense_to_sparse(train_adj)[0]
    t_dist = torch.norm(t_hidden[col]-t_hidden[row], p=2, dim=-1)
    t_dist = torch.exp(-1/2 * t_dist)
    t_dist = to_dense_adj(edge_index=dense_to_sparse(train_adj)[0], edge_attr=t_dist).view(N,N).to(device)

    s_dist = torch.norm(s_hidden[col]-s_hidden[row], p=2, dim=-1)
    s_dist = torch.exp(-1/2 * s_dist)
    s_dist = to_dense_adj(edge_index=dense_to_sparse(train_adj)[0], edge_attr=s_dist).view(N,N).to(device)

    for i in range(N):
        ls = s_dist[i][train_adj[i].to(torch.bool)]
        lt = t_dist[i][train_adj[i].to(torch.bool)]
        if(len(ls)>1):
            ls = F.log_softmax(ls, dim=0)
            lt = F.softmax(lt, dim=0)
            LS.append(kl_op_batch(ls, lt).item())
    LSP_loss = np.mean(LS)

    # final_loss
    loss_train = loss_CE + lbd_LSP*LSP_loss
    loss_train.backward()
    optimizer.step()

    return loss_train.item(),acc_train.item()


def validate():
    teacher.eval()
    model.eval()

    with torch.no_grad():
        s_output, s_hidden = model(features,adj)
        t_output, t_hidden = teacher(features,adj)
        loss_CE = F.nll_loss(s_output[idx_val], labels[idx_val].to(device))

        N = t_hidden[idx_val].size()[0]
        t_dist = torch.zeros(t_hidden.size()[0], t_hidden.size()[0])
        s_dist = torch.zeros(s_hidden.size()[0], s_hidden.size()[0])

        LS = list()
        val_adj = adj.to_dense()
        val_adj = val_adj[idx_val][:, idx_val]
        row, col = dense_to_sparse(val_adj)[0]
        t_dist = torch.norm(t_hidden[col]-t_hidden[row], p=2, dim=-1)
        t_dist = torch.exp(-1/2 * t_dist)
        t_dist = to_dense_adj(edge_index=dense_to_sparse(val_adj)[0], edge_attr=t_dist).view(N,N).to(device)

        s_dist = torch.norm(s_hidden[col]-s_hidden[row], p=2, dim=-1)
        s_dist = torch.exp(-1/2 * s_dist)
        s_dist = to_dense_adj(edge_index=dense_to_sparse(val_adj)[0], edge_attr=s_dist).view(N,N).to(device)

        for i in range(N):
            ls = s_dist[i][val_adj[i].to(torch.bool)]
            lt = t_dist[i][val_adj[i].to(torch.bool)]
            if(len(ls)>1):
                ls = F.log_softmax(ls, dim=0)
                lt = F.softmax(lt, dim=0)
                LS.append(kl_op_batch(ls, lt).item())
        LSP_loss = np.mean(LS)
        loss_val = loss_CE + lbd_LSP*LSP_loss
        acc_val = accuracy(s_output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output,_ = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()

t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra,acc_tra = train()
    loss_val,acc_val = validate()
    """
    if(epoch+1)%1 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            'acc:{:.2f}'.format(acc_tra*100),
            '| val',
            'loss:{:.3f}'.format(loss_val),
            'acc:{:.2f}'.format(acc_val*100))
    """
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == 200:
        break

if args.test:
    acc = test()[1]

#print(count_params(teacher), count_params(model))
#print("Train cost: {:.4f}s".format(time.time() - t_total))
#print('Load {}th epoch'.format(best_epoch))
#print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
print(acc*100)
