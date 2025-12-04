#encoding=utf-8
import torch
import numpy as np
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import dgl
from Model.model import HeteroRGCN, HeteroPredictionModel
from utils import get_comp_g_edge_labels, get_comp_g_path_labels
from utils import hetero_src_tgt_khop_in_subgraph, eval_edge_mask_auc, eval_edge_mask_topk_path_hit
import pandas as pd
from Utils.loader_utils import define_dataloader

def df_node_remap(drug_combination_df ,node_drug1:dict, node_drug2:dict, node_cell:dict, node_drug12:dict):
    drug_combination_df['drug1_db'] = drug_combination_df['Drug1'].map(node_drug1)
    drug_combination_df['drug2_db'] = drug_combination_df['Drug2'].map(node_drug2)
    drug_combination_df['Drug12'] = drug_combination_df['Drug12'].map(node_drug12)
    drug_combination_df['cell'] = drug_combination_df['cell'].map(node_cell)
    drug_combination = drug_combination_df[['drug1_db', 'drug2_db', 'Drug12', 'cell', 'synergistic']]
    return drug_combination

def read_dict(file_path):
    f_read = open(file_path, 'rb')
    dict = pickle.load(f_read)
    f_read.close()
    return dict

parser = argparse.ArgumentParser(description='Interpret drug synergy')
parser.add_argument('--device_id', type=int, default=-1)
'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='./Data/graph')
parser.add_argument('--dataset_name', type=str, default='graph_drugcombdb')
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--test_ratio', type=float, default=0.2)

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=64)
parser.add_argument('--saved_model_dir', type=str, default='../saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Synergy predictor args
'''
parser.add_argument('--src_ntype', type=str, default='drug12', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='cell', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='treats', help='prediction edge type')

'''
Explanation args
'''
parser.add_argument('--num_hops', type=int, default=5, help='computation graph number of hops')
parser.add_argument('--saved_explanation_dir', type=str, default='../saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--eval_explainer_names', nargs='+', default=['SDCInterpreter'],
                    help='name of explainers to evaluate')
parser.add_argument('--eval_path_hit', default=True, action='store_true',
                    help='Whether to save the explanation')
parser.add_argument('--results_file', type=str, default='../results/eval_explanations.txt',
                    help='saving directory of results file')
args = parser.parse_args()
device = torch.device('cpu')

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_' + str(1) + '_model'

pred_pair_to_edge_labels = torch.load('../datasets/graph_DrugCombDB_pred_pair_to_edge_labels')
pred_pair_to_path_labels = torch.load('../datasets/graph_DrugCombDB_pred_pair_to_path_labels')
graph_list, _ = dgl.load_graphs('../Data/graph/graph_drugcombdb')
mp_g = graph_list[0]
mp_g = mp_g.to(device)

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroPredictionModel(encoder, args.src_ntype, args.tgt_ntype)

state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)

test = pd.read_csv('../Data/CV_test_data_symbol.csv')
synergy_pos = test[test.iloc[:, 4] == 1]

test_val = synergy_pos[['Drug1', 'Drug2', 'Drug12', 'cell', 'synergistic']]
nodes_path = '../Data/graph/nodes/'
node_drug1 = read_dict(nodes_path + 'drug1_dict.pkl')
node_drug2 = read_dict(nodes_path + 'drug2_dict.pkl')
node_drug12 = read_dict(nodes_path + 'drug12_dict.pkl')
node_cell = read_dict(nodes_path + 'cell_dict.pkl')
test_val_pos = df_node_remap(test_val, node_drug1, node_drug2, node_cell, node_drug12)
test_src = test_val_pos.iloc[:, 2].to_numpy()  # src
test_tgt = test_val_pos.iloc[:, 3].to_numpy()  # tgt
test_drug1, test_drug2 = test_val_pos.iloc[:, 0].to_numpy(), test_val_pos.iloc[:, 1].to_numpy()

test_val_pos = np.array(test_val_pos).astype(int)
testLoader = define_dataloader(synergy=test_val_pos, batch_size=1, train=False)

test_src = torch.from_numpy(test_src).to(device)
# test_src_10 = test_src[:1000]
test_tgt = torch.from_numpy(test_tgt).to(device)
test_drug1, test_drug2 = torch.from_numpy(test_drug1).to(device), torch.from_numpy(test_drug2).to(device)

comp_graphs = defaultdict(list)
comp_g_labels = defaultdict(list)
test_ids = range(test_src.shape[0])

for i in tqdm(test_ids):
    # Get the k-hop subgraph
    src_nid, tgt_nid = test_src[i], test_tgt[i]
    drug1_nid, drug2_nid = test_drug1[i], test_drug2[i]
    comp_g_src_nid, comp_g_tgt_nid, comp_g_drug1_nid, comp_g_drug2_nid, comp_g, comp_g_feat_nids = hetero_src_tgt_khop_in_subgraph(args.src_ntype,
                                                                                                                                   src_nid,
                                                                                                                                   args.tgt_ntype,
                                                                                                                                   tgt_nid,
                                                                                                                                   drug1_nid,
                                                                                                                                   drug2_nid,
                                                                                                                                   mp_g,
                                                                                                                                   args.num_hops)
    with torch.no_grad():
        index = torch.stack((comp_g_drug1_nid, comp_g_drug2_nid, comp_g_tgt_nid)).transpose(0, 1)
        pred = model(comp_g, index, comp_g_feat_nids).sigmoid().item() > 0.5

    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        comp_graphs[src_tgt] = [comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids]

        # Get labels with subgraph nids and eids
        if src_tgt not in pred_pair_to_edge_labels:
            next
        else:
            edge_labels = pred_pair_to_edge_labels[src_tgt]
            comp_g_edge_labels = get_comp_g_edge_labels(comp_g, edge_labels)

            path_labels = pred_pair_to_path_labels[src_tgt]
            comp_g_path_labels = get_comp_g_path_labels(comp_g, path_labels)

            comp_g_labels[src_tgt] = [comp_g_edge_labels, comp_g_path_labels]

explanation_masks = {}
for explainer in args.eval_explainer_names:
    saved_explanation_mask = f'{explainer}_graph_drugcombdb_1_model_pred_edge_to_comp_g_edge_mask'
    saved_file = Path.cwd().joinpath(args.saved_explanation_dir, saved_explanation_mask)
    with open(saved_file, "rb") as f:
        explanation_masks[explainer] = pickle.load(f)

for explainer in args.eval_explainer_names:
    print(explainer)
    print('-'*30)
    pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

    mask_auc_list = []
    mask_recall_list = []
    cm_list = []
    for src_tgt in comp_graphs:
        if src_tgt in comp_g_labels:
        # comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids = comp_graphs[src_tgt]
            comp_g_edge_labels, comp_g_path_labels = comp_g_labels[src_tgt]
            if src_tgt in pred_edge_to_comp_g_edge_mask:
                comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
                mask_auc, mask_recall, mask_cm = eval_edge_mask_auc(comp_g_edge_mask_dict, comp_g_edge_labels)
                if mask_auc >= 0:
                    mask_auc_list += [mask_auc]
                mask_recall_list += [mask_recall]
                cm_list += [mask_cm]

    avg_auc = np.mean(mask_auc_list)
    avg_recall = np.mean(mask_recall_list)

    sum_cm = np.zeros((2, 2))
    for cm in cm_list:
        sum_cm += cm
    avg_cm = sum_cm / len(cm_list)

    # Print
    np.set_printoptions(precision=4, suppress=True)
    print(f'Average Mask-AUC: {avg_auc : .4f}')
    print(f'Average Mask-Recall: {avg_recall : .4f}')
    print(f'Average Mask-CM: {avg_cm}')

    print('-' * 30, '\n')

if args.eval_path_hit:
    topks = [3, 5, 10, 20, 50, 100, 200]
    for explainer in args.eval_explainer_names:
        print(explainer)
        print('-'*30)
        pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

        explainer_to_topk_path_hit = defaultdict(list)
        for src_tgt in comp_graphs:
            if len(comp_g_labels[src_tgt]) > 0:
                comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
                if src_tgt in comp_g_labels:
                    comp_g_path_labels = comp_g_labels[src_tgt][1]
                    if src_tgt in pred_edge_to_comp_g_edge_mask:
                        comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
                        topk_to_path_hit = eval_edge_mask_topk_path_hit(comp_g_edge_mask_dict, comp_g_path_labels, topks)

                        for topk in topk_to_path_hit:
                            explainer_to_topk_path_hit[topk] += [topk_to_path_hit[topk]]

        # Take average
        explainer_to_topk_path_hit_rate = defaultdict(list)
        for topk in explainer_to_topk_path_hit:
            metric = np.array(explainer_to_topk_path_hit[topk])
            explainer_to_topk_path_hit_rate[topk] = metric.mean(0)

        # Print
        np.set_printoptions(precision=4, suppress=True)
        for k, hr in explainer_to_topk_path_hit_rate.items():
            print(f'k: {k :3} | Path HR: {hr.item(): .4f}')

        print('-'*30, '\n')





