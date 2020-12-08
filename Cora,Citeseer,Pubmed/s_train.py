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
parser.add_argument('--lbd_kl', type=float, default=0)
parser.add_argument('--lbd_pr', type=float, default=0)
parser.add_argument('--kernel', default='kl', help='kl,lin,poly,dist,RBF')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

lbd_kl = args.lbd_kl
lbd_pr = args.lbd_pr
# Load data
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
t_PATH = "./teacher/teacher_"+str(args.data)+str(args.layer)+".pth"
checkpt_file = "./student/student_"+str(args.data)+str(args.layer)+str(lbd_kl)+str(lbd_pr)+str(args.kernel)+".pth"

teacher = GCNII(nfeat=features.shape[1],
                        nlayers=args.layer,
                        nhidden=args.hidden,
                        nclass=int(labels.max()) + 1,
                        dropout=args.dropout,
                        lamda = args.lamda,
                        alpha=args.alpha,
                        variant=args.variant).to(device)
teacher.load_state_dict(torch.load(t_PATH))
model = GCNII_student(nfeat=features.shape[1],
                nlayers=args.layer,
                thidden=args.hidden,
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
#print(count_params(model))

kl_loss_op = torch.nn.KLDivLoss(reduction='none')
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

    # loss_task
    t_output = t_output / temperature
    t_y = F.softmax(t_output[idx_train], dim=1)
    s_y = F.log_softmax(s_output[idx_train], dim=1)
    loss_task = kl_loss_op(s_y, t_y)
    loss_task = torch.mean(torch.sum(loss_task, dim=1))

    # loss_hidden
    t_x = t_hidden[idx_train]
    s_x = s_hidden[idx_train]
    loss_hidden = kernel(t_x, s_x, args.kernel)

    #print(loss_CE, loss_task, loss_hidden)
    # final_loss
    loss_train = loss_CE + lbd_kl*loss_task + lbd_pr*loss_hidden
    loss_train.backward()
    optimizer.step()

    return loss_train.item(),acc_train.item()


def validate():
    teacher.eval()
    model.eval()

    with torch.no_grad():
        t_output, t_hidden = teacher(features,adj)
        s_output, s_hidden = model(features,adj)
        loss_CE = F.nll_loss(s_output[idx_val], labels[idx_val].to(device))

        # loss_task
        t_output = t_output / temperature
        t_y = F.softmax(t_output[idx_val], dim=1)
        s_y = F.log_softmax(s_output[idx_val], dim=1)
        loss_task = kl_loss_op(s_y, t_y)
        loss_task = torch.mean(torch.sum(loss_task, dim=1))

        # loss_hidden
        t_x = t_hidden[idx_val]
        s_x = s_hidden[idx_val]
        loss_hidden = kernel(t_x, s_x, args.kernel)

        loss_val = loss_CE + lbd_kl*loss_task + lbd_pr*loss_hidden
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
    if(epoch+1)%1 == 0:
        """
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
