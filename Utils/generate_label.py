import pickle
import torch
import pandas as pd
from collections import defaultdict
import dgl

def file_read(file):
    with open(file) as f:
        data = f.readlines()
    return data

def save_dict(dict, file_path):
    f_save = open(file_path, 'wb')
    pickle.dump(dict, f_save)
    f_save.close()

def read_dict(file_path):
    f_read = open(file_path, 'rb')
    dict = pickle.load(f_read)
    f_read.close()
    return dict

nodes_path = '../Data/graph/nodes/'
node_drug12 = read_dict(nodes_path + 'drug12_dict.pkl')
node_drug1 = read_dict(nodes_path + 'drug1_dict.pkl')
node_drug2 = read_dict(nodes_path + 'drug2_dict.pkl')
node_gene = read_dict(nodes_path + 'gene_dict.pkl')
node_pathway = read_dict(nodes_path + 'pathway_dict.pkl')
node_cell = read_dict(nodes_path + 'cell_dict.pkl')

drug12_drug1_file = '../Data/dict/drug1_drug12.pkl'
drug12_drug1_dict = read_dict(drug12_drug1_file)

drug12_drug2_file = '../Data/dict/drug2_drug12.pkl'
drug12_drug2_dict = read_dict(drug12_drug2_file)

drug1_gene_file = '../Data/dict/drug1_gene.pkl'
drug1_gene = read_dict(drug1_gene_file)

drug2_gene_file = '../Data/dict/drug2_gene.pkl'
drug2_gene = read_dict(drug2_gene_file)

cell_gene_file = '../Data/dict/cell_gene.pkl'
cell_gene = read_dict(cell_gene_file)

gene_pathway_file = '../Data/dict/gene_pathway.pkl'
gene_pathway_dict = read_dict(gene_pathway_file)

drug12_dg1 = ('drug12', 'contain1', 'drug1')
drug12_dg2 = ('drug12', 'contain2', 'drug2')
# drug12_dg = ('drug12', 'contains', 'single_drug')
dg1_gene = ('drug1', 'aims-at1', 'gene1')
dg2_gene = ('drug2', 'aims-at2', 'gene1')
# dg_gene = ('drug', 'aims-at', 'gene1')
# drug12_drug = ('drug12', 'contain', 'single-drug')
# drug_gene = ('drug', 'aims-at', 'gene')
gene_p = ('gene1', 'included-by', 'pathway')
p_gene = ('pathway', 'includes', 'gene2')
gene_ce = ('gene2', 'causes', 'cell')

path_label_dict = defaultdict(list)
edge_label_dict = {}

drug_pair2single = pd.read_csv('../Data/DrugCombDB/drug_combinations_treated.csv')
drug_pair2single = drug_pair2single[['drug1_db', 'drug2_db', 'Drug12']]
drug_combination = pd.read_csv('../Data/DrugCombDB/drug_combinations.csv')
drug_combination = drug_combination[['Drug12', 'cell', 'synergistic']]
drug_combination = drug_combination[drug_combination['synergistic'] == 1].reset_index(drop=True)
drug_combination = drug_combination[['Drug12', "cell"]]
drug_combination = drug_combination.groupby('Drug12')['cell'].agg(lambda x: list(x)).reset_index()
drug_treat_cell_dict = dict(zip(drug_combination.iloc[:, 0], drug_combination.iloc[:, 1]))


for drug12, cell in drug_treat_cell_dict.items():
    drug1 = drug_pair2single[drug_pair2single['Drug12'] == drug12][['drug1_db']].iloc[0, 0]
    drug2 = drug_pair2single[drug_pair2single['Drug12'] == drug12][['drug2_db']].iloc[0, 0]
    for c in cell:
        gt_pair = (('drug12', node_drug12[drug12]), ('cell', node_cell[c]))
        if gt_pair not in edge_label_dict:
            edge_label_dict[gt_pair] = defaultdict(set)
            edge_label_dict[gt_pair][drug12_dg1] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][drug12_dg2] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][dg1_gene] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][dg2_gene] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][gene_p] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][p_gene] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
            edge_label_dict[gt_pair][gene_ce] = (
                torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))

        gene_drug1 = drug1_gene[drug1]
        gene_drug2 = drug2_gene[drug2]
        if c in cell_gene:
            gene_cell = cell_gene[c]
        else:
            gene_cell = []

        #drug_combination-single_drug-gene-pathway-gene-cell
        #drug1-cell
        for gene_d1 in gene_drug1:
            if gene_d1 in gene_pathway_dict:
                pathway_set1 = gene_pathway_dict[gene_d1]
            else:
                pathway_set1 = []
            for gene_c in gene_cell:
                if gene_c in gene_pathway_dict:
                    pathway_set3 = gene_pathway_dict[gene_c]
                else:
                    pathway_set3 = []
                intersection = set(pathway_set1) & set(pathway_set3)

                for pathway in intersection:
                    e1 = (torch.tensor(node_drug12[drug12]), torch.tensor(node_drug1[drug1]))
                    e2 = (torch.tensor(node_drug1[drug1]), torch.tensor(node_gene[gene_d1]))
                    e3 = (torch.tensor(node_gene[gene_d1]), torch.tensor(node_pathway[pathway]))
                    e4 = (torch.tensor(node_pathway[pathway]), torch.tensor(node_gene[gene_c]))
                    e5 = (torch.tensor(node_gene[gene_c]), torch.tensor(node_cell[c]))

                    p1 = (drug12_dg1, node_drug12[drug12], node_drug1[drug1])
                    p2 = (dg1_gene, node_drug1[drug1], node_gene[gene_d1])
                    p3 = (gene_p, node_gene[gene_d1], node_pathway[pathway])
                    p4 = (p_gene, node_pathway[pathway], node_gene[gene_c])
                    p5 = (gene_ce, node_gene[gene_c], node_cell[c])
                    path = [p1, p2, p3, p4, p5]
                    if path not in path_label_dict[gt_pair]:
                        path_label_dict[gt_pair].append(path)
                        # edge_label_dict[gt_pair][dg2tg]
                        src, dst = edge_label_dict[gt_pair][drug12_dg1]
                        src = torch.cat((src, torch.tensor([e1[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e1[1]])), dim=0)
                        edge_label_dict[gt_pair][drug12_dg1] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][dg1_gene]
                        src = torch.cat((src, torch.tensor([e2[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e2[1]])), dim=0)
                        edge_label_dict[gt_pair][dg1_gene] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][gene_p]
                        src = torch.cat((src, torch.tensor([e3[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e3[1]])), dim=0)
                        edge_label_dict[gt_pair][gene_p] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][p_gene]
                        src = torch.cat((src, torch.tensor([e4[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e4[1]])), dim=0)
                        edge_label_dict[gt_pair][p_gene] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][gene_ce]
                        src = torch.cat((src, torch.tensor([e5[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e5[1]])), dim=0)
                        edge_label_dict[gt_pair][gene_ce] = (src.to(torch.int64), dst.to(torch.int64))


        #drug2-cell
        for gene_d2 in gene_drug2:
            if gene_d2 in gene_pathway_dict:
                pathway_set2 = gene_pathway_dict[gene_d2]
            else:
                pathway_set2 = []
            for gene_c in gene_cell:
                if gene_c in gene_pathway_dict:
                    pathway_set3 = gene_pathway_dict[gene_c]
                else:
                    pathway_set3 = []
                intersection = set(pathway_set2) & set(pathway_set3)

                for pathway in intersection:
                    e1 = (torch.tensor(node_drug12[drug12]), torch.tensor(node_drug2[drug2]))
                    e2 = (torch.tensor(node_drug2[drug2]), torch.tensor(node_gene[gene_d2]))
                    e3 = (torch.tensor(node_gene[gene_d2]), torch.tensor(node_pathway[pathway]))
                    e4 = (torch.tensor(node_pathway[pathway]), torch.tensor(node_gene[gene_c]))
                    e5 = (torch.tensor(node_gene[gene_c]), torch.tensor(node_cell[c]))

                    p1 = (drug12_dg2, node_drug12[drug12], node_drug2[drug2])
                    p2 = (dg2_gene, node_drug2[drug2], node_gene[gene_d2])
                    p3 = (gene_p, node_gene[gene_d2], node_pathway[pathway])
                    p4 = (p_gene, node_pathway[pathway], node_gene[gene_c])
                    p5 = (gene_ce, node_gene[gene_c], node_cell[c])
                    path = [p1, p2, p3, p4, p5]
                    if path not in path_label_dict[gt_pair]:
                        path_label_dict[gt_pair].append(path)
                        # edge_label_dict[gt_pair][dg2tg]
                        src, dst = edge_label_dict[gt_pair][drug12_dg2]
                        src = torch.cat((src, torch.tensor([e1[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e1[1]])), dim=0)
                        edge_label_dict[gt_pair][drug12_dg2] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][dg2_gene]
                        src = torch.cat((src, torch.tensor([e2[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e2[1]])), dim=0)
                        edge_label_dict[gt_pair][dg2_gene] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][gene_p]
                        src = torch.cat((src, torch.tensor([e3[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e3[1]])), dim=0)
                        edge_label_dict[gt_pair][gene_p] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][p_gene]
                        src = torch.cat((src, torch.tensor([e4[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e4[1]])), dim=0)
                        edge_label_dict[gt_pair][p_gene] = (src.to(torch.int64), dst.to(torch.int64))
                        #
                        src, dst = edge_label_dict[gt_pair][gene_ce]
                        src = torch.cat((src, torch.tensor([e5[0]])), dim=0)
                        dst = torch.cat((dst, torch.tensor([e5[1]])), dim=0)
                        edge_label_dict[gt_pair][gene_ce] = (src.to(torch.int64), dst.to(torch.int64))
        if gt_pair not in path_label_dict.keys():
            edge_label_dict.pop(gt_pair, None)

torch.save(path_label_dict, '../datasets/graph_DrugCombDB_pred_pair_to_path_labels')
torch.save(edge_label_dict, '../datasets/graph_DrugCombDB_pred_pair_to_edge_labels')
print('end')