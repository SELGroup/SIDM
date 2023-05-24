#!/usr/bin/env python
# encoding: utf-8
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("lib/")
from mlp import MLP

class TreeCNN(nn.Module):
    # v2.1 在分层结构的基础上，删除输入层的mlp
    def __init__(self, depth, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, tree_pooling_type, device):
        '''
        depth: the depth of coding trees (EXCLUDING the leaf layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the leaf nodes)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        tree_pooling_type: how to aggregate entire nodes in a tree (root, sum, mean)
        device: which device to use
        '''
        super(TreeCNN, self).__init__()
        self.final_dropout = final_dropout
        self.device = device
        self.depth = depth
        self.tree_pooling_type = tree_pooling_type

        ###List of MLPs
        self.mlps = torch.nn.ModuleList([None])
        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList([None])
        for layer in range(1, self.depth+1):
            if layer == 1:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.depth+1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

    def __preprocess_children_sumpool(self, batch_tree):
        ### 每层根据和下层的连接来生成自己的表征
        Adj_blocks = [None]  # jump leaf layer
        for layer in range(1, self.depth+1):
            edge_mat_list = []
            start_pdx = [0]  # 上一层的节点统计
            start_idx = [0]
            for i, tree in enumerate(batch_tree):
                start_pdx.append(start_pdx[i] + tree['node_size'][layer-1])
                start_idx.append(start_idx[i] + tree['node_size'][layer])
                edge_mat_list.append(torch.LongTensor(tree['edges'][layer]) \
                                    + torch.LongTensor([start_idx[i], start_pdx[i]]))

            Adj_block_idx = torch.cat(edge_mat_list, 0).transpose(0, 1)
            Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
            Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem,
                                                 torch.Size([start_idx[-1], start_pdx[-1]]))
            Adj_blocks.append(Adj_block.to(self.device))
        return Adj_blocks

    def __preprocess_treepool(self, batch_tree):
        ### create sum or average pooling sparse matrix over entire nodes in each tree (num trees x num nodes)
        pools = []
        # 根节点只有一个，不需要readout
        for layer in range(self.depth):
            start_idx = [0]
            #compute the padded neighbor list
            for i, tree in enumerate(batch_tree):
                start_idx.append(start_idx[i] + tree['node_size'][layer])

            idx = []
            elem = []
            for i, tree in enumerate(batch_tree):
                node_size = tree['node_size'][layer]
                ### average pooling
                if self.tree_pooling_type == "average":
                    elem.extend([1. / node_size] * node_size)
                ### sum pooling
                else:
                    elem.extend([1] * node_size)
                idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1])])
            elem = torch.FloatTensor(elem)
            idx = torch.LongTensor(idx).transpose(0, 1)
            tree_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_tree), start_idx[-1]]))
            pools.append(tree_pool.to(self.device))
        return pools

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        # pooling children nodes
        pooled = torch.spmm(Adj_block, h)
        hn = self.mlps[layer](pooled)
        hn = self.batch_norms[layer](hn)
        # non-linearity
        hn = F.relu(hn)
        return hn


    def forward(self, batch_tree):
        # node_feature: (node_size, node_embedding)
        X_concat = torch.cat([tree['node_features'] for tree in batch_tree], 0).to(self.device)
        pools = self.__preprocess_treepool(batch_tree)

        h = X_concat
        h_rep = [h]

        # 分层的传播矩阵
        Adj_blocks = self.__preprocess_children_sumpool(batch_tree)
        # 逐层收敛
        for layer in range(1, self.depth+1):
            h = self.next_layer(h, layer, Adj_block = Adj_blocks[layer])
            h_rep.append(h)
            # print(layer, h)

        score_over_layer = 0
        # batch_size * hidden_size  tree_embedding
        if self.tree_pooling_type != 'root':
            for layer in range(self.depth):
                pooled_h = torch.spmm(pools[layer], h_rep[layer])
                score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        # root pool, 根节点只有一个
        score = F.dropout(self.linears_prediction[self.depth](h), self.final_dropout, training=self.training)
        score_over_layer += score

        return score_over_layer
