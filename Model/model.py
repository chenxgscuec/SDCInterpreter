import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import HeteroEmbedding, EdgePredictor

'''
HeteroRGCN model adapted from the DGL official tutorial
https://docs.dgl.ai/en/0.6.x/tutorials/basics/5_hetero.html
https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/4_rgcn.html
'''
class Mlp(nn.Module):
    def __init__(self, kg_size, class_num):
        super(Mlp, self).__init__()
        num_hidden = kg_size * 3

        act = nn.ReLU()
        dropout = nn.Dropout(0.5)
        self.fc_layer = nn.Sequential(nn.Linear(num_hidden, num_hidden // 2),
                                      # nn.BatchNorm1d(num_hidden // 2),
                                      nn.LayerNorm(num_hidden // 2),
                                      act,
                                      dropout,

                                      nn.Linear(num_hidden // 2, num_hidden // 4),
                                      nn.LayerNorm(num_hidden // 4),
                                      act,
                                      dropout,

                                      nn.Linear(num_hidden // 4, num_hidden // 8),
                                      nn.LayerNorm(num_hidden // 8),
                                      act,
                                      dropout
                                      )
        self.output_layer = nn.Sequential(nn.Linear(num_hidden // 8, class_num, bias=False))

    def forward(self, *embs):
        input = torch.cat((embs), dim=-1)
        pred = self.fc_layer(input)
        pred = self.output_layer(pred)
        return torch.sigmoid(pred.squeeze(dim=1))

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_0 for transform the node's own feature
        # self.emb_dim = in_size
        self.weight0 = nn.Linear(in_size, out_size)
        
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, g, feat_dict, eweight_dict=None):
        # The input is a dictionary of node features for each type
        funcs = {}
        if eweight_dict is not None:
            # Store the sigmoid of edge weights
            g.edata['_edge_weight'] = eweight_dict
                
        for ntype in g.ntypes:
            # Compute h_0 = W_0 * h
            h0 = self.weight0(feat_dict[ntype])
            g.nodes[ntype].data['h0'] = h0  
            g.nodes[ntype].data['h'] = h0         
            # g.nodes[ntype].data['h'] = torch.empty((g.number_of_nodes(ntype), self.emb_dim))
        
        for srctype, etype, dsttype in g.canonical_etypes:
            # Compute h_0 = W_0 * h
            # h0 = self.weight0(feat_dict[srctype])
            # g.nodes[srctype].data['h0'] = h0
            
            # Compute h_r = W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph_1 for message passing
            g.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            if eweight_dict is not None:
                msg_fn = fn.u_mul_e('Wh_%s' % etype, '_edge_weight', 'm')
            else:
                msg_fn = fn.copy_u('Wh_%s' % etype, 'm')
                
            funcs[(srctype, etype, dsttype)] = (msg_fn, fn.mean('m', 'h'))

        def apply_func(nodes):
            h = nodes.data['h'] + nodes.data['h0']
            # h = nodes.data.get('h', torch.empty_like(nodes.data['h0'])) + nodes.data['h0']
            return {'h': h}
        
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",

        # "min", "mean", "stack"
        g.multi_update_all(funcs, 'sum', apply_func)
        # g.multi_update_all(funcs, 'sum')

        # return the updated node feature dictionary
        return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, g, emb_dim, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        self.emb = HeteroEmbedding({ntype : g.num_nodes(ntype) for ntype in g.ntypes}, emb_dim)
        self.eweight = None
        self.layer1 = HeteroRGCNLayer(emb_dim, hidden_size, g.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, g.etypes)

    def forward(self, g, feat_nids=None, eweight_dict=None):
        if feat_nids is None:
            feat_dict = self.emb.weight
        else:
            feat_dict = self.emb(feat_nids)
        h_dict = self.layer1(g, feat_dict, eweight_dict)
        # h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        # h_dict = self.layer2(g, h_dict, eweight_dict)
        return h_dict


class HeteroPredictionModel(nn.Module):
    def __init__(self, encoder, src_ntype, tgt_ntype):
        super().__init__()
        self.encoder = encoder
        self.decoder = Mlp(64, class_num=1)
        self.src_ntype = src_ntype
        self.tgt_ntype = tgt_ntype

    def encode(self, g, feat_nids=None, eweight_dict=None):
        h = self.encoder(g, feat_nids, eweight_dict)
        return h

    # def forward(self,src1, src2, tgt, g, index, feat_nids=None, eweight_dict=None, page=False):
    def forward(self, g, index, feat_nids=None, eweight_dict=None):
        h = self.encode(g, feat_nids, eweight_dict)
        drug1 = h['drug1'][index[:, 0]]
        drug2 = h['drug2'][index[:, 1]]
        cell = h['cell'][index[:, 2]]
        pred1 = self.decoder(drug1, drug2, cell)
        return pred1



