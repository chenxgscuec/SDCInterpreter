#encoding=utf-8
import os
import argparse
import pickle
import dgl
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from Model.Interpreter import PaGELink
from Utils.utils import set_seed
from Model.model import HeteroRGCN, HeteroPredictionModel
from Utils.loader_utils import *

def read_dict(file_path):
    f_read = open(file_path, 'rb')
    dict = pickle.load(f_read)
    f_read.close()
    return dict

parser = argparse.ArgumentParser(description='Interpret drug synergy mechanism')
parser.add_argument('--device_id', type=int, default=0)

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
parser.add_argument('--saved_model_dir', type=str, default='./saved_models')
parser.add_argument('--saved_model_name', type=str, default='')
parser.add_argument('--src_ntype', type=str, default='drug12', help='prediction source node type')
parser.add_argument('--tgt_ntype', type=str, default='cell', help='prediction target node type')

'''
Explanation args
'''
parser.add_argument('--lr', type=float, default=0.01, help='explainer learning_rate')
parser.add_argument('--alpha', type=float, default=0.01, help='explainer on-path edge regularizer weight')
parser.add_argument('--beta', type=float, default=0.25, help='explainer off-path edge regularizer weight')
parser.add_argument('--num_hops', type=int, default=5, help='computation graph_1 number of hops')
parser.add_argument('--num_epochs', type=int, default=50, help='How many epochs to learn the mask')
parser.add_argument('--num_paths', type=int, default=200, help='How many paths to generate')
parser.add_argument('--max_path_length', type=int, default=5, help='max length of generated paths')
parser.add_argument('--k_core', type=int, default=1, help='k for the k-core graph_1')
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph_1 such that all nodes have degree smaller than max_degree. No prune if -1')
parser.add_argument('--save_explanation', default=True, action='store_true',
                    help='Whether to save the explanation')
parser.add_argument('--saved_explanation_dir', type=str, default='./saved_explanations',
                    help='directory of saved explanations')
args = parser.parse_args()
set_seed(0)

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_' + str(1) + '_model'

graph_list, _ = dgl.load_graphs('Data/graph/graph_drugcombdb')
mp_g = graph_list[0].to(device)

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroPredictionModel(encoder, args.src_ntype, args.tgt_ntype)
state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)

SDCInterpreter = PaGELink(model,
                    lr=args.lr,
                    alpha=args.alpha,
                    beta=args.beta,
                    num_epochs=args.num_epochs,
                    log=True).to(device)


test_data = pd.read_csv('./Data/CV_test_data' + '.csv', header=None, names=['drug1', 'drug2', 'Drug12', 'cell', 'synergy'])
test_src = test_data.iloc[:, 2].to_numpy()#Drug12
test_tgt = test_data.iloc[:, 3].to_numpy()#cell
test_drug1, test_drug2 = test_data.iloc[:, 0].to_numpy(), test_data.iloc[:, 1].to_numpy()#drug1,drug2

test_val = test_data[['drug1', 'drug2', 'cell', 'synergy']]
test_val = np.array(test_val).astype(int)
testLoader = define_dataloader(synergy=test_val, batch_size=1, train=False)

test_tgt = torch.tensor(test_tgt).to(device)
test_src = torch.tensor(test_src).to(device)
test_drug1, test_drug2 = torch.tensor(test_drug1).to(device), torch.tensor(test_drug2).to(device)
test_ids = range(test_src.shape[0])

pred_edge_to_comp_g_edge_mask = {}
pred_edge_to_paths = {}


for i in tqdm(test_ids):
    src_nid, tgt_nid = test_src[i].unsqueeze(0), test_tgt[i].unsqueeze(0)
    drug1_nid, drug2_nid = test_drug1[i].unsqueeze(0), test_drug2[i].unsqueeze(0)
    with torch.no_grad():
        index = torch.stack((drug1_nid, drug2_nid, tgt_nid)).transpose(0, 1)
        pred = model(mp_g, index.to(device)).item() > 0.5
    if pred:
        src_tgt = ((args.src_ntype, int(src_nid)), (args.tgt_ntype, int(tgt_nid)))
        paths, comp_g_edge_mask_dict = SDCInterpreter.explain(src_nid,
                                                        tgt_nid,
                                                        drug1_nid,
                                                        drug2_nid,
                                                        mp_g,
                                                        args.num_hops,
                                                        args.prune_max_degree,
                                                        args.k_core,
                                                        args.num_paths,
                                                        args.max_path_length,
                                                        prune_graph=True,
                                                        return_mask=True)

        pred_edge_to_comp_g_edge_mask[src_tgt] = comp_g_edge_mask_dict
        pred_edge_to_paths[src_tgt] = paths


if args.save_explanation:
    if not os.path.exists(args.saved_explanation_dir):
        os.makedirs(args.saved_explanation_dir)

    saved_edge_explanation_file = f'SDCInterpreter_{args.saved_model_name}_pred_edge_to_comp_g_edge_mask'
    saved_path_explanation_file = f'SDCInterpreter_{args.saved_model_name}_pred_edge_to_paths'
    pred_edge_to_comp_g_edge_mask = {edge: {k: v.cpu() for k, v in mask.items()} for edge, mask in
                                     pred_edge_to_comp_g_edge_mask.items()}

    saved_edge_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_edge_explanation_file)
    with open(saved_edge_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_comp_g_edge_mask, f)

    saved_path_explanation_path = Path.cwd().joinpath(args.saved_explanation_dir, saved_path_explanation_file)
    with open(saved_path_explanation_path, "wb") as f:
        pickle.dump(pred_edge_to_paths, f)
