#encoding=utf-8
import pickle
import pandas as pd
import inspect
from dgl.data.utils import save_graphs
import numpy as np
import dgl

# save file path
nodes_path = '../Data/graph/nodes/'

def read_file(file):
    with open(file, encoding='utf-8', errors='ignore') as f:
        data = f.readlines()
    return data

def read_dict(file_path):
    f_read = open(file_path, 'rb')
    dict = pickle.load(f_read)
    f_read.close()
    return dict

def save_dict(dict, file_path):
    f_save = open(file_path, 'wb')
    pickle.dump(dict, f_save)
    f_save.close()

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

## nodes
drug2 = []
drug1 = []
drug12 = []
cell = []
gene = []
pathway = []

drug1_drug12_file = '../Data/dict/drug1_drug12.pkl'
drug1_drug12_dict = read_dict(drug1_drug12_file)
for k, v in drug1_drug12_dict.items():
    if k not in drug1:
        drug1.append(k)
    for d in v:
        if d not in drug12:
            drug12.append(d)

drug2_drug12_file = '../Data/dict/drug2_drug12.pkl'
drug2_drug12_dict = read_dict(drug2_drug12_file)
for k, v in drug2_drug12_dict.items():
    if k not in drug2:
        drug2.append(k)
    for d in v:
        if d not in drug12:
            drug12.append(d)


drug1_gene_file = '../Data/dict/drug1_gene.pkl'
drug1_gene_dict = read_dict(drug1_gene_file)
for k, v in drug1_gene_dict.items():
    if k not in drug1:
        drug1.append(k)
    for g in v:
        if g not in gene:
            gene.append(g)

drug2_gene_file = '../Data/dict/drug2_gene.pkl'
drug2_gene_dict = read_dict(drug2_gene_file)
for k, v in drug2_gene_dict.items():
    if k not in drug2:
        drug2.append(k)
    for g in v:
        if g not in gene:
            gene.append(g)

gene_cell_file = '../Data/dict/gene_cell.pkl'
gene_cell_dict = read_dict(gene_cell_file)
for k,v in gene_cell_dict.items():
    if k not in gene:
        gene.append(k)
    for c in v:
        if c not in cell:
            cell.append(c)

gene_pathway_file = '../Data/dict/gene_pathway.pkl'
gene_pathway_dict = read_dict(gene_pathway_file)
for k, v in gene_pathway_dict.items():
    if k not in gene:
        gene.append(k)
    for p in v:
        if p not in pathway:
            pathway.append(p)

drug1_dict, drug2_dict, drug12_dict, cell_dict, gene_dict, pathway_dict = {}, {}, {}, {}, {}, {}
node_list = [drug1, drug2, drug12, cell, gene, pathway]
dict_list = [drug1_dict, drug2_dict, drug12_dict, cell_dict, gene_dict, pathway_dict]

for index in range(len(node_list)):
    nodes = node_list[index]
    dic = dict_list[index]
    var_name = retrieve_name(dic)[0]
    for i in range(len(nodes)):
        dic[nodes[i]] = i
    save_path = nodes_path + var_name + '.pkl'
    save_dict(dic, save_path)

##edges
drug1_drug12_file = '../Data/dict/drug1_drug12.pkl'
drug2_drug12_file = '../Data/dict/drug2_drug12.pkl'
drug1_gene_file = '../Data/dict/drug1_gene.pkl'
drug2_gene_file = '../Data/dict/drug2_gene.pkl'
# drug12_cell_file = '../Data/dict/drug12_cell.pkl'
gene_cell_file = '../Data/dict/gene_cell.pkl'
gene_pathway_file = '../Data/dict/gene_pathway.pkl'

node_drug1 = read_dict('../Data/graph/nodes/drug1_dict.pkl')
node_drug2 = read_dict('../Data/graph/nodes/drug2_dict.pkl')
node_drug12 = read_dict('../Data/graph/nodes/drug12_dict.pkl')
node_cell = read_dict('../Data/graph/nodes/cell_dict.pkl')
node_gene = read_dict('../Data/graph/nodes/gene_dict.pkl')
node_pathway = read_dict('../Data/graph/nodes/pathway_dict.pkl')

drug1_drug12_dict = read_dict(drug1_drug12_file)
d1_2_d12_adj = np.zeros((len(node_drug1), len(node_drug12)))
for key in drug1_drug12_dict:
    drug1 = key
    u = node_drug1[drug1]
    for drug12 in drug1_drug12_dict[key]:
        v = node_drug12[drug12]
        d1_2_d12_adj[u][v] = 1

drug2_drug12_dict = read_dict(drug2_drug12_file)
d2_2_d12_adj = np.zeros((len(node_drug2), len(node_drug12)))
for key in drug2_drug12_dict:
    drug2 = key
    u = node_drug2[drug2]
    for drug12 in drug2_drug12_dict[key]:
        v = node_drug12[drug12]
        d2_2_d12_adj[u][v] = 1

drug1_gene_dict = read_dict(drug1_gene_file)
d1_2_g_adj = np.zeros((len(node_drug1), len(node_gene)))
for key in drug1_gene_dict:
    drug1 = key
    u = node_drug1[drug1]
    for gene in drug1_gene_dict[key]:
        v = node_gene[gene]
        d1_2_g_adj[u][v] = 1

drug2_gene_dict = read_dict(drug2_gene_file)
d2_2_g_adj = np.zeros((len(node_drug2), len(node_gene)))
for key in drug2_gene_dict:
    drug2 = key
    u = node_drug2[drug2]
    for gene in drug2_gene_dict[key]:
        v = node_gene[gene]
        d2_2_g_adj[u][v] = 1

# drug12_cell_dict = read_dict(drug12_cell_file)
# d12_2_c_adj = np.zeros((len(node_drug12),len(node_gene)))
# for key in drug12_cell_dict:
#     drug12 = key
#     u = node_drug12[drug12]
#     for cell in drug12_cell_dict[key]:
#         v = node_cell[cell]
#         d12_2_c_adj[u][v] = 1

gene_cell_dict = read_dict(gene_cell_file)
g_2_c_adj = np.zeros((len(node_gene), len(node_cell)))
for key in gene_cell_dict:
    gene = key
    u = node_gene[gene]
    for cell in gene_cell_dict[key]:
        v = node_cell[cell]
        g_2_c_adj[u][v] = 1

gene_pathway_dict = read_dict(gene_pathway_file)
g_2_p_adj = np.zeros((len(node_gene), len(node_pathway)))
for key in gene_pathway_dict:
    gene = key
    u = node_gene[gene]
    for pathway in gene_pathway_dict[key]:
        v = node_pathway[pathway]
        g_2_p_adj[u][v] = 1


data = {"D1vsD12": d1_2_d12_adj, "D2vsD12": d2_2_d12_adj, "D1vsG": d1_2_g_adj, "D2vsG": d2_2_g_adj, "GvsC": g_2_c_adj,  "GvsP": g_2_p_adj}
G = dgl.heterograph({
        ('drug12', 'contain1', 'drug1'): data['D1vsD12'].transpose().nonzero(),
        ('drug12', 'contain2', 'drug2'): data['D2vsD12'].transpose().nonzero(),
        ('drug1', 'aims-at1', 'gene1'): data['D1vsG'].nonzero(),
        ('drug2', 'aims-at2', 'gene1'): data['D2vsG'].nonzero(),
        # ('drug12','treat','cell'):data['D12vsC'].nonzero(),
        ('gene2', 'causes', 'cell'): data['GvsC'].nonzero(),
        ('gene1', 'included-by', 'pathway'): data['GvsP'].nonzero(),
        ('pathway', 'includes', 'gene2'): data['GvsP'].transpose().nonzero(),
    },
    num_nodes_dict = {'drug1': len(node_drug1), 'drug2': len(node_drug2),
                      'drug12': len(node_drug12), 'cell': len(node_cell),
                      'gene1':len(node_gene), 'gene2': len(node_gene), 'pathway': len(node_pathway)}
    )
save_graphs(f'../Data/graph/graph_drugcombdb', G)