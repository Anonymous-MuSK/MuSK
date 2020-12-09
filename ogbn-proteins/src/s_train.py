import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_softmax
from torch_geometric.nn.conv import MessagePassing
from utils import *
from model import *


dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeSampler(data, num_parts=40, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)

gpu_id = 3
nl = 28
ver = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
t_model = DeeperGCN(hidden_channels=64, num_layers=nl).to(device)
s_model = student(hidden_channels=64, num_layers=2).to(device)
t_optimizer = torch.optim.Adam(t_model.parameters(), lr=0.01)
s_optimizer = torch.optim.Adam(s_model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')


def train(epoch):
    t_model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        t_optimizer.zero_grad()
        data = data.to(device)
        out, _ = t_model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        t_optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


def train_student_model(t_model, epoch):
    kl_loss_op = torch.nn.KLDivLoss(reduction='none')
    temperature = 2
    lbd_pred = 0.1
    lbd_embd = 0.01

    s_model.train()
    t_model.eval()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        s_optimizer.zero_grad()
        data = data.to(device)

        # loss_pred
        s_logits, s_xs = s_model(data.x, data.edge_index, data.edge_attr)
        t_logits, t_xs = t_model(data.x, data.edge_index, data.edge_attr)
        t_logits = t_logits / temperature
        s_y = F.log_softmax(s_logits[data.train_mask], dim=1)
        t_y = F.softmax(t_logits[data.train_mask], dim=1)
        pred_loss = kl_loss_op(s_y, t_y)
        pred_loss = torch.mean(torch.sum(kl_loss, dim=1))

        # loss_BCE
        BCE_loss = criterion(s_logits[data.train_mask], data.y[data.train_mask])

        # loss_embd
        t_x = F.softmax(t_xs[-1][data.train_mask], dim=1)
        s_x = F.log_softmax(s_xs[-1][data.train_mask], dim=1)
        embd_loss = torch.mean(torch.sum(kl_loss_op(s_x, t_x), dim=1))

        # loss_final
        loss = BCE_loss + lbd_pred*pred_loss + lbd_embd*embd_loss
        loss.backward()
        s_optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out, _ = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


t_max_idx = 0
s_max_idx = 0
t_max_acc = [0, 0, 0]
s_max_acc = [0, 0, 0]
t_PATH = "./baseline_models/teacher" + str(nl) + '_' + str(ver) + ".pth"
s_PATH = "./ourkd_models/student" + str(nl) + '_' + str(ver) + ".pth"

t_model.load_state_dict(torch.load(t_PATH))
t_model.eval()
for epoch in range(1, 1001):
    if(epoch==1):
        t_max_acc = test(t_model)
        print(f'[Teacher] Train: {t_max_acc[0]:.4f}, '
            f'Val: {t_max_acc[1]:.4f}, Test: {t_max_acc[2]:.4f}')

    loss = train_student_model(t_model, epoch)
    train_rocauc, valid_rocauc, test_rocauc = test(s_model)
    if(valid_rocauc>s_max_acc[1]):
        s_max_acc[0] = train_rocauc
        s_max_acc[1] = valid_rocauc
        s_max_acc[2] = test_rocauc
        torch.save(s_model.state_dict(), s_PATH)
    print(f'[Student] Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')

print(f'[Teacher] #Parameters: {count_parameters(t_model):02d}, Train: {t_max_acc[0]:.4f}, '
    f'Val: {t_max_acc[1]:.4f}, Test: {t_max_acc[2]:.4f}')
print(f'[Student] #Parameters: {count_parameters(s_model):02d}, Train: {s_max_acc[0]:.4f}, '
    f'Val: {s_max_acc[1]:.4f}, Test: {s_max_acc[2]:.4f}')
