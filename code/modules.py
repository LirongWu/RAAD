import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F


def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class RelationMPNN(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, dropout=0.1, edges_in_d=1, edge_type=8):
        super(RelationMPNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.coord_mlp = nn.ModuleList()
        self.relation_mlp = nn.ModuleList()

        self.message_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + n_channel**2 + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU())

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf + edges_in_d + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, edges_in_d))
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf))
         
        for _ in range(edge_type):
            self.relation_mlp.append(nn.Linear(input_nf, input_nf, bias=False))

            layer = nn.Linear(hidden_nf, n_channel, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            self.coord_mlp.append(nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.SiLU(),
                layer
            ))

    def message_model(self, source, target, radial, edge_attr):
        radial = radial.reshape(radial.shape[0], -1)
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.message_mlp(out)
        out = self.dropout(out)

        return out

    def node_model(self, x, edge_list, edge_feat_list, sampled_index_list):
        agg = self.relation_mlp[0](unsorted_segment_sum(edge_feat_list[0], edge_list[0][0], num_segments=x.size(0)))
        for i in range(1, len(edge_list)):
            if i == 6 and (sampled_index_list is not None):
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i][sampled_index_list[0]], edge_list[i][0][sampled_index_list[0]], num_segments=x.size(0)))
            elif i == 7 and (sampled_index_list is not None):
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i][sampled_index_list[1]], edge_list[i][0][sampled_index_list[1]], num_segments=x.size(0)))
            else:
                agg += self.relation_mlp[i](unsorted_segment_sum(edge_feat_list[i], edge_list[i][0], num_segments=x.size(0)))
            
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        out = self.dropout(out)
        out = x + out

        return out
    
    def coord_model(self, coord, edge_list, edge_feat_list, coord_diff_list, segment_ids):
        tran_list = []
        row_list = []
        if segment_ids is None:
            sampled_index_list = None
        else:
            sampled_index_list = []

        for i in range(len(edge_list)):
            trans = coord_diff_list[i] * self.coord_mlp[i](edge_feat_list[i]).unsqueeze(-1)  # [n_edge, n_channel, d]
            edges = edge_list[i][0]
            if (i == 6 or i == 7) and (segment_ids is not None):
                antigen_edge_list = sequential_or(segment_ids[edge_list[i][0]] == 3, segment_ids[edge_list[i][1]] == 3)
                sampled_index = torch.ones(trans.shape[0]).to(trans.device)
                if antigen_edge_list.sum() != 0:
                    weight = torch.abs(self.coord_mlp[i](edge_feat_list[i]).mean(dim=-1))[antigen_edge_list]
                    sampled_index[antigen_edge_list] = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=0.5, probs=(weight  - weight.min()) / (weight.max() - weight.min())).rsample()
                sampled_index = sampled_index.bool()
                trans = trans[sampled_index]
                edges = edge_list[i][0][sampled_index]
                sampled_index_list.append(sampled_index)
            tran_list.append(trans)
            row_list.append(edges)
        agg = unsorted_segment_mean(torch.cat(tran_list, dim=0), torch.cat(row_list, dim=0), num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        coord = coord + agg

        return coord, sampled_index_list

    def edge_model(self, h, edge_list, edge_feat_list):
        m = []

        for i in range(len(edge_list)):
            row, col = edge_list[i]
            out = torch.cat([h[row], edge_feat_list[i], h[col]], dim=1)
            out = self.edge_mlp(out)
            m.append(out)

        return m
    
    def forward(self, h, coord, edge_attr, edge_list, segment_ids=None):

        edge_feat_list = []
        coord_diff_list = []

        for i in range(len(edge_list)):
            radial, coord_diff = coord2radial(edge_list[i], coord)
            coord_diff_list.append(coord_diff)

            row, col = edge_list[i]
            edge_feat = self.message_model(h[row], h[col], radial, edge_attr[i])
            edge_feat_list.append(edge_feat)

        x, sampled_index_list = self.coord_model(coord, edge_list, edge_feat_list, coord_diff_list, segment_ids)
        h = self.node_model(h, edge_list, edge_feat_list, sampled_index_list)
        m = self.edge_model(h, edge_list, edge_attr)

        return h, x, m


class RelationEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, n_layers=4, dropout=0.1, node_feats_dim=0, edge_feats_dim=1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf + node_feats_dim, hidden_nf)
        self.linear_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(f'layer_{i}', RelationMPNN(hidden_nf, hidden_nf, hidden_nf, n_channel, dropout=dropout, edges_in_d=edge_feats_dim))
    
    def forward(self, h, x, edges_list, edge_feats_list, node_feats, segment_ids, interface_only):
        h = torch.cat((h, node_feats), 1)
        h = self.linear_in(h)
        h = self.dropout(h)

        m = edge_feats_list
        for i in range(self.n_layers):
            if interface_only == 0:
                h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list)
            else:
                h, x, m = self._modules[f'layer_{i}'](h, x, m, edges_list, segment_ids)

        out = self.dropout(h)
        out = self.linear_out(out)

        return out, x, h