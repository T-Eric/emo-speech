import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

from src.utils import tools
from src.models.gcn import GCN
from src.models.gat import EnhancedSkipGCNGAT, SkipGCNGAT
import src.config as config

crit = nn.CrossEntropyLoss(weight=torch.ones(8, device='cuda'))  # unweighted


def train(args, model: nn.Module, device, train_data, optimizer, epoch, A):
    # train_data: [[feature],[label]]
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    start = time.time()

    for pos in pbar:
        selected_idx = np.random.permutation(len(train_data))[:args.batch_size]
        batch_data = [train_data[idx] for idx in selected_idx]
        X_concat = torch.stack(
            [torch.tensor(data[0], dtype=torch.float32)for data in batch_data]).to(device)
        labels = torch.tensor(
            [data[1] for data in batch_data], dtype=torch.long).to(device)

        output = model(X_concat)
        loss = crit(output, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        if pos > 0:
            pbar.set_description(
                f'Epoch: {epoch}, Loss: {loss_accum / pos:.4f}')

    end = time.time()
    average_loss = loss_accum/total_iters
    print('Epoch: %d, Loss: %.4f, Time: %.4f' %
          (epoch, average_loss, end-start))
    return average_loss


def pass_data_iteratively(model: nn.Module, data, minibatch_size=64):
    '''get the result of the model on the data iteratively'''
    model.eval()
    output = []
    idx = np.arange(len(data))
    for i in range(0, len(data), minibatch_size):
        sample_idx = idx[i:i+minibatch_size]
        if len(sample_idx) == 0:
            continue
        X_concat = torch.stack([torch.tensor(data[j][0], dtype=torch.float32)
                                for j in sample_idx]).to(next(model.parameters()).device)
        output.append(model(X_concat).detach())
    return torch.cat(output, dim=0)


def test(args, model: nn.Module, train_data, test_data):
    model.eval()

    train_output = pass_data_iteratively(model, train_data)
    train_pred = train_output.max(1, keepdim=True)[1]
    train_labels = torch.LongTensor(
        [data[1] for data in train_data]).to(device)
    train_correct = train_pred.eq(
        train_labels.view_as(train_pred)).sum().cpu().item()
    train_acc = train_correct/float(len(train_data))

    test_output = pass_data_iteratively(model, test_data)
    test_pred = test_output.max(1, keepdim=True)[1]
    test_labels = torch.LongTensor([data[1] for data in test_data]).to(device)
    test_correct = test_pred.eq(
        test_labels.view_as(test_pred)).sum().cpu().item()
    test_acc = test_correct/float(len(test_data))

    print('Train Acc: %.4f, Test Acc: %.4f' % (train_acc, test_acc))
    return train_acc, test_acc, test_output, test_labels


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Emotion Recognition Competition for GNNs!')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 42)')
    parser.add_argument('--fold_idx', type=int, default=5,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'gat', 'exgat'], help='which model to use (default: gcn)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers INCLUDING the input one (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--layer_dropout', type=float, default=0.1,
                        help='middle layer dropout (default: 0.1, exgat only)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling over nodes in a graph to get graph embeddig: sum, average or max (default: sum)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping. Default 10.')
    parser.add_argument('--adjacency', default=1,
                        type=int, help='Nearest neighbors per node. Default 1.')
    parser.add_argument('--self_connect', default=False, type=bool,
                        help='Whether to connect itself in the Adjacency Mat. Default False.')
    # exgat only
    parser.add_argument('--use_multiscale_fusion', default=True, type=bool)
    parser.add_argument('--use_special_pooling', default=True, type=bool)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda:'+str(args.device)
                          if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_data = {
        'epochs': [[] for _ in range(args.fold_idx)],
        'train_losses': [[] for _ in range(args.fold_idx)],
        'val_losses': [[] for _ in range(args.fold_idx)],
        'train_accs': [[] for _ in range(args.fold_idx)],
        'test_accs': [[] for _ in range(args.fold_idx)],
        'folds': args.fold_idx,
        'model_config': vars(args)  # 保存模型配置
    }

    data, n, d = tools.load_data()
    num_classes = config.NUM_CLASSES
    train_folds, test_folds = tools.separate_data(
        data, args.seed, args.fold_idx)
    A = torch.Tensor(tools.adj_builder(n, adj_num=args.adjacency,
                     self_connect=args.self_connect)).to(device)
    gat_heads = args.adjacency*2

    acc_train_sum = 0
    acc_test_sum = 0

    for i in range(args.fold_idx):
        if args.model == 'gcn':
            # GCN
            model = GCN(args.num_layers, d, args.hidden_dim, num_classes,
                        args.final_dropout, args.graph_pooling_type, device, A).to(device)
        elif args.model == 'gat':
            # SkipGCNGAT，头数采用adjacency数，hidden_layer尽量比输入维数大
            model = SkipGCNGAT(args.num_layers, d, args.hidden_dim, num_classes, gat_heads,
                               args.final_dropout, args.graph_pooling_type, device, A).to(device)
        elif args.model == 'exgat':
            # EnhancedSkipGCNGAT，头数采用adjacency数，hidden_layer尽量比输入维数大
            model = EnhancedSkipGCNGAT(args.num_layers, d, args.hidden_dim, num_classes, gat_heads,
                                       args.final_dropout, args.graph_pooling_type, device, A, args.layer_dropout,args.use_multiscale_fusion, args.use_special_pooling).to(device)

        train_data = train_folds[i]
        test_data = test_folds[i]

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, step_size=50, gamma=0.5)
        # 换成余弦退火
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5)

        early_stopping = tools.EarlyStopping(
            patience=args.patience, verbose=True)

        for epoch in range(1, args.epochs+1):
            avg_loss = train(args, model, device,
                             train_data, optimizer, epoch, A)
            log_data['epochs'][i].append(epoch)
            log_data['train_losses'][i].append(float(avg_loss))

            if epoch > 1:
                with torch.no_grad():
                    val_out = pass_data_iteratively(model, test_data)
                    val_labels = torch.LongTensor(
                        [data[1] for data in test_data]).to(device)
                    val_loss = np.average(
                        crit(val_out, val_labels).detach().cpu().numpy())
                    log_data['val_losses'][i].append(val_loss)

                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print('Early stopping')
                    break

            if epoch % 10 == 0:
                acc_train, acc_test, _, _ = test(
                    args, model, train_data, test_data)
                log_data['train_accs'][i].append(float(acc_train))
                log_data['test_accs'][i].append(float(acc_test))

            scheduler.step()

        model.load_state_dict(torch.load('checkpoint.pt'))

        acc_train, acc_test, output, label = test(
            args, model, train_data, test_data)
        acc_train_sum += acc_train
        acc_test_sum += acc_test

    print('Average train acc: %f,  Average test acc: %f' %
          (acc_train_sum / args.fold_idx, acc_test_sum / args.fold_idx))

    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f'logs/{args.model}_l{args.num_layers}_h{args.hidden_dim}_{timestamp}.json'
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4, cls=NumpyEncoder)
