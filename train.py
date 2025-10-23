#encoding=utf-8
import argparse
import numpy as np
from Utils.utils import metrics_graph, set_seed_all
import glob
import os
from sklearn.model_selection import KFold, train_test_split
from Utils.utils import Data
from Utils.loader_utils import *
from Model.model import HeteroRGCN, HeteroPredictionModel
import pandas as pd

def training(model, graph, optimizer, trainLoader):
    loss_train = 0
    true_ls, pre_ls = [], []
    optimizer.zero_grad()
    for batch, (pair, label) in enumerate(trainLoader):
        pred = model(g=graph, index=pair.to(device))
        loss = loss_func(pred, label.to(device).to(torch.float32))
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        pre_ls += pred.cpu().detach().numpy().tolist()
        true_ls += label.cpu().detach().numpy().tolist()
    auc_train, aupr_train, f1_train, acc_train = metrics_graph(true_ls, pre_ls)
    return [auc_train, aupr_train, f1_train, acc_train], loss_train

def test(model, graph, validLoader):
    model.eval()
    true_ls, pre_ls = [], []
    with torch.no_grad():
        for batch, (pair, label) in enumerate(validLoader):
            pred = model(g=graph, index=pair.to(device))
            loss = loss_func(pred, label.to(device).to(torch.float32))
            pre_ls += pred.cpu().detach().numpy().tolist()
            true_ls += label.cpu().detach().numpy().tolist()
        auc_test, aupr_test, f1_test, acc_test = metrics_graph(label.cpu().detach().numpy(),
                                                               pred.cpu().detach().numpy())
    return [auc_test, aupr_test, f1_test, acc_test], loss.item(), pred.cpu().detach().numpy()

def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[4] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[4] == 0])
    # -----split synergy into 5CV,test set
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    # --CV set
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # --test set
    synergy_test = np.concatenate((np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)
    np.random.shuffle(synergy_cv_data)
    np.random.shuffle(synergy_test)
    np.savetxt(path + 'test_y_true.txt', synergy_test[:, 4])
    return synergy_cv_data, synergy_test


def parse_args():
    parser = argparse.ArgumentParser(description='Training and Testing Knowledge Graph Embedding Models')
    parser.add_argument('--cuda', default='cuda:0', action='store_true', help='use GPU')
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay of adam")
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save the model')
    parser.add_argument('--saved_model_dir', type=str, default='./saved_models/', help='Where to save the model')
    parser.add_argument('--dataset_name', type=str, default='graph_drugcombdb')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=768)
    return parser.parse_args()

if torch.cuda.is_available() :
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

args = parse_args()
net_data = Data()
graph = net_data.graph_all.to(device)
synergy_data = net_data.drug_synergy
seed = 0
cv_mode_ls = [1, 2, 3]


for cv_mode in cv_mode_ls:
    path = f'./results/' + args.dataset_name + '_' + str(cv_mode) + '_'
    file = open(path + 'result.txt', 'w')
    set_seed_all(seed)
    synergy_cv, index_test = data_split(synergy_data)
    index_test_df = pd.DataFrame(index_test)
    index_test_df.to_csv('./Data/CV_test_data.csv', index=False, header=False)
    synergy_cv, synergy_test = np.delete(synergy_cv, 2, axis=1), np.delete(index_test, 2, axis=1)
    if cv_mode == 1:
        cv_data = synergy_cv
    elif cv_mode == 2:
        cv_data = np.unique(synergy_cv[:, 2])
    else:
        cv_data = np.unique(np.vstack([synergy_cv[:, 0], synergy_cv[:, 1]]), axis=1).T
    final_metric = np.zeros(4)
    fold_num = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in kf.split(cv_data):
        # ---construct train_set+validation_set
        if cv_mode == 1:
            synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
        elif cv_mode == 2:
            train_name, test_name = cv_data[train_index], cv_data[validation_index]
            synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
            synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
        else:
            pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
            synergy_train = np.array(
                [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            synergy_validation = np.array(
                [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
        np.savetxt(path + 'val_' + str(fold_num+1) + '_true.txt', synergy_validation[:, 3])
        # --DataLoader
        trainLoader = define_dataloader(synergy=synergy_train, batch_size=args.batch_size, train=True)
        validLoader = define_dataloader(synergy=synergy_validation, batch_size=args.batch_size, train=True)
        testLoader = define_dataloader(synergy=synergy_test, batch_size=args.batch_size, train=False)

        # --model_build
        encoder = HeteroRGCN(graph, args.emb_dim, args.hidden_dim, args.out_dim)
        model = HeteroPredictionModel(encoder, src_ntype='', tgt_ntype='').to(device)
        loss_func = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # --run
        best_metric = [0, 0, 0, 0]
        best_epoch = 0
        for epoch in range(args.num_epoch):
            model.train()
            train_metric, train_loss = training(model, graph, optimizer, trainLoader)
            val_metric, val_loss, _ = test(model, graph, validLoader)
            if epoch % 20 == 0:
                print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                      'AUC: {:.6f},'.format(train_metric[0]), 'AUPR: {:.6f},'.format(train_metric[1]),
                      'F1: {:.6f},'.format(train_metric[2]), 'ACC: {:.6f},'.format(train_metric[3]),
                      )
                print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                      'AUC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                      'F1: {:.6f},'.format(val_metric[2]), 'ACC: {:.6f},'.format(val_metric[3]))
            torch.save(model.state_dict(), '{}.pth'.format(epoch))
            if val_metric[0] > best_metric[0]:
                best_metric = val_metric
                best_epoch = epoch
            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(f)
        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
              'AUC: {:.6f},'.format(best_metric[0]),
              'AUPR: {:.6f},'.format(best_metric[1]), 'F1: {:.6f},'.format(best_metric[2]),
              'ACC: {:.6f},'.format(best_metric[3]))
        model.load_state_dict(torch.load('{}.pth'.format(best_epoch)))
        val_metric, _, y_val_pred = test(model, graph, validLoader)
        test_metric, _, y_test_pred = test(model, graph, testLoader)
        np.savetxt(path + 'val_' + str(fold_num+1) + '_pred.txt', y_val_pred)
        np.savetxt(path + 'test_' + str(fold_num+1) + '_pred.txt', y_test_pred)
        file.write('val_metric:')
        for item in val_metric:
            file.write(str(item) + '\t')
        file.write('\ntest_metric:')
        for item in test_metric:
            file.write(str(item) + '\t')
        file.write('\n')
        final_metric += test_metric
        if args.save_model:
            output_dir = args.saved_model_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), output_dir + str(cv_mode) + '_' + f"{args.dataset_name}_" + str(fold_num+1) +"_model.pth")
            print('--- saving model successfully. ---')
        fold_num = fold_num + 1
    final_metric /= 5
    files = glob.glob('*.pth')
    for f in files:
        os.remove(f)
    print('Final 5-cv average results, AUC: {:.6f},'.format(final_metric[0]),
          'AUPR: {:.6f},'.format(final_metric[1]),
          'F1: {:.6f},'.format(final_metric[2]), 'ACC: {:.6f},'.format(final_metric[3]))

