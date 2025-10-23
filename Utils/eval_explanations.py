#encoding=utf-8
import torch
import numpy as np
import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm
import sys
import dgl
from Utils.data_processing import load_dataset
from Model.model import HeteroRGCN, HeteroPredictionModel
from Utils.utils import set_config_args, get_comp_g_edge_labels, get_comp_g_path_labels
from Utils.utils import hetero_src_tgt_khop_in_subgraph, eval_edge_mask_auc, eval_edge_mask_topk_path_hit
import pandas as pd
from Utils.loader_utils import define_dataloader

parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=0)
'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='./Data/graph')
parser.add_argument('--dataset_name', type=str, default='graph_drugcombdb')
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--max_num_samples', type=int, default=-1, 
                    help='maximum number of samples to explain, for fast testing. Use all if -1')

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--out_dim', type=int, default=64)
# parser.add_argument('--saved_model_dir', type=str, default='saved_models')
parser.add_argument('--saved_model_dir', type=str, default='../saved_models')
parser.add_argument('--saved_model_name', type=str, default='')

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='drug12', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='cell', help='prediction target node type')
parser.add_argument('--pred_etype', type=str, default='treats', help='prediction edge type')
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                   help='operation passed to dgl.EdgePredictor')


'''
Explanation args
'''
parser.add_argument('--num_hops', type=int, default=3, help='computation graph number of hops')
# parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
#                     help='directory of saved explanations')
parser.add_argument('--saved_explanation_dir', type=str, default='./saved_explanations',
                    help='directory of saved explanations')
parser.add_argument('--eval_explainer_names', nargs='+', default=['pagelink'],#'gnnexp','pgexp'],
                    help='name of explainers to evaluate') 
parser.add_argument('--eval_path_hit', default=True, action='store_true', 
                    help='Whether to save the explanation') 
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')
parser.add_argument('--results_file', type=str, default='../results/eval_explanations_2.txt',
                    help='saving directory of results file')

args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'train_eval')

if 'HERB' in args.dataset_name:
    args.src_ntype = 'herb'
    args.tgt_ntype = 'disease'
    args.pred_etype = 'has-effects-on'
else:
    args.src_ntype = 'drug12'
    args.tgt_ntype = 'cell'
    args.pred_etype = 'treats'

if args.link_pred_op in ['cat']:
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}


device = torch.device('cpu')
pred_pair_to_edge_labels = torch.load(f'../datasets/graph_DrugCombDB_pred_pair_to_edge_labels')
pred_pair_to_path_labels = torch.load(f'../datasets/graph_DrugCombDB_pred_pair_to_path_labels')
graph_list, _ = dgl.load_graphs('../Data/graph/graph_drugcombdb')
mp_g = graph_list[0].to(device)

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroPredictionModel(encoder, args.src_ntype, args.tgt_ntype)

# for i in range(1, 6):
i = 1
if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_' + str(i) + '_model'

state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)

test_data = pd.read_csv('../Data/CV_test_data.csv', header=None, names=['drug1', 'drug2', 'Drug12', 'cell', 'synergy'])
test_ids = range(len(test_data))
test_src = test_data.iloc[:, 2].to_numpy()#Drug12
test_tgt = test_data.iloc[:, 3].to_numpy()#cell
test_drug1, test_drug2 = test_data.iloc[:, 0].to_numpy(),test_data.iloc[:, 1].to_numpy()#drug1,drug2

test_val = test_data[['drug1', 'drug2', 'cell', 'synergy']]
test_val = np.array(test_val).astype(int)
testLoader = define_dataloader(synergy=test_val, batch_size=1, train=False)

test_src = torch.from_numpy(test_src).to(device)#转换为tensor
test_tgt = torch.from_numpy(test_tgt).to(device)
test_drug1, test_drug2 = torch.from_numpy(test_drug1).to(device), torch.from_numpy(test_drug2).to(device)
if args.max_num_samples > 0:
    test_ids = test_ids[:args.max_num_samples]

comp_graphs = defaultdict(list)
comp_g_labels = defaultdict(list)

for batch, (pair, label) in tqdm(enumerate(testLoader), unit="batch"):
    # Get the k-hop subgraph
    src_nid, tgt_nid = test_src[batch].unsqueeze(0), test_tgt[batch].unsqueeze(0)
    drug1_nid, drug2_nid = test_drug1[batch].unsqueeze(0), test_drug2[batch].unsqueeze(0)
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
        comp_g = comp_g.to(device)
        pred = model(comp_g, index, comp_g_feat_nids).sigmoid().item() > 0.5

    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        comp_graphs[src_tgt] = [comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids]

        if src_tgt not in pred_pair_to_edge_labels:
            next
        else:
            edge_labels = pred_pair_to_edge_labels[src_tgt]
            comp_g_edge_labels = get_comp_g_edge_labels(comp_g, edge_labels)

            path_labels = pred_pair_to_path_labels[src_tgt]
            comp_g_path_labels = get_comp_g_path_labels(comp_g, path_labels)

            comp_g_labels[src_tgt] = [comp_g_edge_labels, comp_g_path_labels]

explanation_masks = {}
explainer = 'pagelink'
# saved_file = '../saved_explanations_previous/saved_explanations/pagelink_graph_drugcombdb_1_model_pred_edge_to_comp_g_edge_mask_1'
# saved_file = '../saved_explanations_previous/saved_explanations_tiaodicanshu/pagelink_graph_drugcombdb_1_model_pred_edge_to_comp_g_edge_mask_1'
saved_file = '../saved_explanations/pagelink_graph_drugcombdb_1_model_pred_edge_to_comp_g_edge_mask_1'
with open(saved_file, "rb") as f:
    explanation_masks[explainer] = pickle.load(f)

with open(args.results_file, 'w', encoding='utf-8') as f:
    sys.stdout = f
    print('Dataset:', args.dataset_name)
    for explainer in args.eval_explainer_names:
        print(explainer)
        print('-'*30)
        pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

        mask_auc_list = []
        mask_recall_list = []
        mask_cm_list = []
        for src_tgt in comp_graphs:
            if src_tgt in comp_g_labels:
                if src_tgt in pred_edge_to_comp_g_edge_mask:
                    # comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
                    comp_g_edge_labels, comp_g_path_labels = comp_g_labels[src_tgt]
                    comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
                    mask_auc, mask_recall, mask_cm = eval_edge_mask_auc(comp_g_edge_mask_dict, comp_g_edge_labels)
                    if mask_auc >= 0:
                        mask_auc_list += [mask_auc]
                    mask_recall_list += [mask_recall]
                    mask_cm_list += [mask_cm]

        avg_auc = np.mean(mask_auc_list)
        avg_recall = np.mean(mask_recall_list)

        sum_cm = np.zeros((2, 2))
        for cm in mask_cm_list:
            sum_cm += cm
        avg_cm = sum_cm/len(mask_cm_list)

        # Print
        np.set_printoptions(precision=4, suppress=True)
        print(f'Average Mask-AUC: {avg_auc : .4f}')
        print(f'Average Mask-Recall: {avg_recall : .4f}')
        print(f'Average Mask-CM: {avg_cm}')

        print('-'*30, '\n')

    if args.eval_path_hit:
        topks = [3, 5, 10, 20, 50, 100, 200]
        for explainer in args.eval_explainer_names:
            print(explainer)
            print('-'*30)
            pred_edge_to_comp_g_edge_mask = explanation_masks[explainer]

            explainer_to_topk_path_hit = defaultdict(list)

            for src_tgt in comp_graphs:
                a = len(comp_g_labels[src_tgt])
                if a > 0:
                    if src_tgt in pred_edge_to_comp_g_edge_mask:
                        comp_g_src_nid, comp_g_tgt_nid, comp_g, comp_g_feat_nids, = comp_graphs[src_tgt]
                        comp_g_path_labels = comp_g_labels[src_tgt][1]
                        comp_g_edge_mask_dict = pred_edge_to_comp_g_edge_mask[src_tgt]
                        comp_g_cat_edge_mask = torch.cat([v for v in comp_g_edge_mask_dict.values()])
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

sys.stdout = sys.__stdout__
