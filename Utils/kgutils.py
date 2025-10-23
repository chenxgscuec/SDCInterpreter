import dgl
import torch
from dgl import DGLGraph
import numpy as np

device = torch.device("cuda:0")



def build_graph(args, net_data, idx=None):
    """ Create a DGL graph_1. The graph_1 is bidirectional because RGCN authors use reversed relations.
        This function also generates edge type and normalization factor (reciprocal of node incoming degree)

        Protein-0, Cellline-1, Drug-2
    """
    num_nodes = args.nentity  # 16810
    net = net_data.nets
    n_label = torch.from_numpy(net_data.type_mask).unsqueeze(dim=1).to(device).to(torch.long)
    g = DGLGraph().to(device)
    g.add_nodes(num_nodes)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1).to(device)
    g.ndata.update({'n_id': node_id})  # drug:0~763 cline:764~839 protein:840~16809
    g.ndata.update({'n_label': n_label})  # node type

    ppi, cpi, dpi = np.array(net[0]), np.array(net[1]), np.array(net[2])
    ppi = np.array([[i[0], i[1]] for i in ppi if i[0] != i[1]])  # remove self_loof

    i = 0
    # protein-protein
    # 靶点-靶点类型的边 设置为0
    g.add_edges(ppi[:, 0], ppi[:, 1], {'e_label': i * torch.ones(len(ppi[:, 0]), 1, dtype=torch.long)})
    i += 1
    # celine-protein triu
    # 细胞系-蛋白类型边 设置为1
    g.add_edges(cpi[:, 0], cpi[:, 1], {'e_label': i * torch.ones(len(cpi[:, 0]), 1, dtype=torch.long)})
    i += 1
    # drug-protein
    # 药物-蛋白类型边 设置为2
    g.add_edges(dpi[:, 0], dpi[:, 1], {'e_label': i * torch.ones(len(dpi[:, 0]), 1, dtype=torch.long)})
    num_one_dir_edges = g.number_of_edges()


    i += 1  # 3
    g.add_edges(ppi[:, 1], ppi[:, 0], {'e_label': i * torch.ones(len(ppi[:, 0]), 1, dtype=torch.long)})
    i += 1
    # protein-celine
    # 蛋白类-细胞系型边 设置为4
    g.add_edges(cpi[:, 1], cpi[:, 0], {'e_label': i * torch.ones(len(cpi[:, 0]), 1, dtype=torch.long)})
    i += 1
    # protein-drug
    # 蛋白类-药物型边 设置为5
    g.add_edges(dpi[:, 1], dpi[:, 0], {'e_label': i * torch.ones(len(dpi[:, 0]), 1, dtype=torch.long)})

    i += 1
    # add self-loop 设置为6
    # if args.self_loop == 1:
    g.add_edges(g.nodes(), g.nodes(), {'e_label': i * torch.ones(g.number_of_nodes(), 1, dtype=torch.long)})

    n_edges = g.number_of_edges()
    edge_id = torch.arange(0, n_edges, dtype=torch.long).to(device)
    g.edata['e_id'] = edge_id


    return g, num_one_dir_edges, i + 1, torch.from_numpy(cpi), torch.from_numpy(dpi)