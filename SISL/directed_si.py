# -*- coding: utf-8 -*-
from collections import defaultdict
import logging
from random import Random

import networkx as nx
import math

import numpy as np
from copy import deepcopy

import torch

class TreeNode:
    def __init__(self, ID, partition, pro_matrix, h=0, parent=None, children=None):
        self.ID = ID
        self.partition = partition
        self.h = h
        self.parent = parent
        self.children = children
        self.pro_matrix = pro_matrix
        self.p_value, self.v_value = self.update_pv_value()
    
    def update_pv_value(self):
        p_value, v_value = 0.0, 0.0
        for x_id in self.partition:
            for y_id in range(1, self.pro_matrix.shape[1] + 1):
                v_value += self.pro_matrix[x_id - 1, y_id - 1]
                if y_id not in self.partition:
                    p_value += self.pro_matrix[x_id - 1, y_id - 1]
        assert p_value <= v_value
        return p_value, v_value
    
    def add_children(self, node_id):
        if self.children is None:
            self.children = [node_id]
        else:
            self.children.append(node_id)
    
    def del_children(self, node_id):
        assert self.children is not None
        self.children.remove(node_id)
    
    def set_parent(self, node_id):
        self.parent = node_id

    def caculate_se(self, node_dict):
        if self.parent is None:
            return 0.0
        vp_value = node_dict[self.parent].v_value
        return - self.p_value * math.log(self.v_value / vp_value)
    
    def print_node_information(self):
        print("ID: {}, parent: {}, h: {}, children: {}, partition: {}, p_value: {}, v_value: {}".format(
            self.ID, self.parent, self.h, self.children, self.partition, self.p_value, self.v_value))

class Directed_Structural_Information:
    '''
        带权有向图结构熵计算类：
            1. 构建带权有向图，保证强连通
            2. 计算节点稳态分布
            3. 更新每个节点的p与v值
            4. 初始化编码树结构
            5. 对于目标节点，维持历史兄弟与当前兄弟集合
            6. 遍历历史兄弟节点执行merge操作，更新兄弟集合与对应pv值
            7. 遍历当前兄弟节点执行combine操作，更新兄弟集合与对应pv值
    '''
    def __init__(self, data=None, pt=None, rep_dict=None):
        self.data = data
        self.rep_dict = rep_dict
        pro_graph, paths = self.construct_pro_graph(pt)
        self.pro_graph = self.make_minimally_strongly_connected(pro_graph)
        self.index_vertex_mapping = {index + 1: vertex for index, vertex in enumerate(list(self.pro_graph.nodes()))}  # node_id: cid, vertex_id = node_id - 1
        self.vertex_index_mapping = {vertex: index + 1 for index, vertex in enumerate(list(self.pro_graph.nodes()))}  # cid: node_id
        self.pro_matrix = self.compute_pro_matrix()
        self.node_dict, self.node_number = dict(), 0
        self.node_dict[0] = TreeNode(ID=self.node_number, partition=list(range(1, self.pro_matrix.shape[0] + 1)), pro_matrix=self.pro_matrix, h=0)
        self.node_number += 1
        for _ in range(1, self.pro_matrix.shape[0] + 1):
            node = TreeNode(ID=self.node_number, partition=[self.node_number], pro_matrix=self.pro_matrix, h=1)
            node.set_parent(0)
            self.node_dict[0].add_children(self.node_number)
            self.node_dict[self.node_number] = node
            self.node_number += 1
        self.optimize_tree(max_height=5)
        # self.print_partition_tree()
        self.paths = self.adjust_paths(paths)
        self.stacked_features = torch.stack([torch.stack([self.rep_dict[pid] for pid in path]) for path in self.paths])
        assert self.stacked_features.shape == self.data.shape

    def construct_pro_graph(self, pt):
        pro_graph, paths = nx.DiGraph(), []
        for cid in self.rep_dict.keys():
            pro_graph.add_node(cid)
        for batch_index in range(self.data.shape[0]):
            batch_data = self.data[batch_index]
            path = []
            for index in range(self.data.shape[1] - 1):
                src_id = batch_index * self.data.shape[1] + index
                dst_id = src_id + 1
                src_cid, dst_cid = self.get_community_id(src_id, pt), self.get_community_id(dst_id, pt)
                if pro_graph.has_edge(src_cid, dst_cid):
                    pro_graph[src_cid][dst_cid]['weight'] += 1.0
                else:
                    pro_graph.add_edge(src_cid, dst_cid, weight=1.0)
                path.append(src_cid)
                if index == self.data.shape[1] - 2:
                    path.append(dst_cid)
            paths.append(path)
        # total_weight = sum(data['weight'] for u, v, data in pro_graph.edges(data=True))
        # for u, v, data in pro_graph.edges(data=True):
        #     pro_graph[u][v]['weight'] /= total_weight
        return pro_graph, paths

    def get_community_id(self, nid, pt):
        for cid in self.rep_dict.keys():
            partition = pt.tree_node[cid].partition
            if nid in partition:
                return cid
    
    def make_minimally_strongly_connected(self, graph, weight=0.05):
        # 解析所有的强连通分量
        scc = list(nx.strongly_connected_components(graph))
        if len(scc) == 1:
            # 图已经是强连通的
            return graph
        # 创建一个新的有向图，表示强连通分量
        scc_graph = nx.DiGraph()
        for i in range(len(scc)):
            scc_graph.add_node(i)
        # 添加边来连接SCCs，确保强连通性
        for i in range(len(scc) - 1):
            scc_graph.add_edge(i, i + 1)
        # 确保图是循环的，将最后一个SCC连接到第一个SCC
        scc_graph.add_edge(len(scc) - 1, 0)
        # 将SCC图中的边映射回原图
        for u, v in scc_graph.edges():
            node_u = next(iter(scc[u]))  # 从SCC u中取一个节点
            node_v = next(iter(scc[v]))  # 从SCC v中取一个节点
            graph.add_edge(node_u, node_v, weight=weight)
        assert nx.is_strongly_connected(graph)
        return graph
    
    def compute_pro_matrix(self):
        # 计算稳态分布
        steady_state_distributions = nx.pagerank(self.pro_graph)
        # print(steady_state_distributions)
        # 构造转移概率矩阵
        adj_matrix = nx.to_numpy_array(self.pro_graph, weight='weight', dtype=float)
        normalized_matrix = self.normalize_rows(adj_matrix)
        for i in range(normalized_matrix.shape[0]):
            normalized_matrix[i, :] = steady_state_distributions[self.index_vertex_mapping[i + 1]] * normalized_matrix[i, :]
        return normalized_matrix

    def normalize_rows(self, matrix):
        """
        将矩阵的每一行规范化，使得每行元素之和为1。
        """
        row_sums = matrix.sum(axis=1)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        return normalized_matrix
    
    def print_partition_tree(self):
        for nid in range(self.node_number):
            if self.node_dict[nid] is None:
                print(nid, ': None')
            else:
                self.node_dict[nid].print_node_information()
    
    def cal_merge_ed(self, src, dst):
        src_node, dst_node = self.node_dict[src], self.node_dict[dst]
        parent_node = self.node_dict[src_node.parent]
        assert (src_node.parent == dst_node.parent) and (src_node.h == dst_node.h) and (src_node.children is not None) and (dst_node.children is not None)
        vp_value, pp_value = src_node.v_value + dst_node.v_value, src_node.p_value + dst_node.p_value
        pp_reduction = 0.0
        for src_pid in src_node.partition:
            for dst_pid in dst_node.partition:
                pp_reduction += (self.pro_matrix[src_pid - 1, dst_pid - 1] + self.pro_matrix[dst_pid - 1, src_pid - 1])
        pp_value -= pp_reduction
        se_red = src_node.caculate_se(self.node_dict) + dst_node.caculate_se(self.node_dict) + pp_value * math.log(vp_value / parent_node.v_value)
        for cid in src_node.children:
            child_node = self.node_dict[cid]
            se_red += child_node.caculate_se(self.node_dict) + child_node.p_value * math.log(child_node.v_value / vp_value)
        for cid in dst_node.children:
            child_node = self.node_dict[cid]
            se_red += child_node.caculate_se(self.node_dict) + child_node.p_value * math.log(child_node.v_value / vp_value)
        return se_red
    
    def cal_combine_ed(self, src, dst):
        src_node, dst_node = self.node_dict[src], self.node_dict[dst]
        parent_node = self.node_dict[src_node.parent]
        assert (src_node.parent == dst_node.parent) and (src_node.h == dst_node.h)
        vp_value, pp_value = src_node.v_value + dst_node.v_value, src_node.p_value + dst_node.p_value
        pp_reduction = 0.0
        for src_pid in src_node.partition:
            for dst_pid in dst_node.partition:
                pp_reduction += (self.pro_matrix[src_pid - 1, dst_pid - 1] + self.pro_matrix[dst_pid - 1, src_pid - 1])
        pp_value -= pp_reduction
        se_reduction = pp_value * math.log(vp_value / parent_node.v_value)
        se_reduction += (src_node.caculate_se(self.node_dict) - src_node.p_value * math.log(src_node.v_value / vp_value)) + (dst_node.caculate_se(self.node_dict) - dst_node.p_value * math.log(dst_node.v_value / vp_value))
        return se_reduction
        
    def exe_merge_op(self, src, dst):
        # print('merging', src, dst)
        src_node, dst_node = self.node_dict[src], self.node_dict[dst]
        parent_node = self.node_dict[src_node.parent]
        assert (src_node.parent == dst_node.parent) and (src_node.h == dst_node.h) and (src_node.children is not None) and (dst_node.children is not None)
        parent_node.del_children(dst)
        for cid in dst_node.children:
            src_node.add_children(cid)
            self.node_dict[cid].set_parent(src)
        src_node.partition += dst_node.partition
        vp_value, pp_value = src_node.v_value + dst_node.v_value, src_node.p_value + dst_node.p_value
        pp_reduction = 0.0
        for src_pid in src_node.partition:
            for dst_pid in dst_node.partition:
                pp_reduction += (self.pro_matrix[src_pid - 1, dst_pid - 1] + self.pro_matrix[dst_pid - 1, src_pid - 1])
        pp_value -= pp_reduction
        src_node.v_value = vp_value
        src_node.p_value = pp_value
        self.node_dict[dst] = None
    
    def exe_combine_op(self, src, dst):
        # print('combining', src, dst)
        src_node, dst_node = self.node_dict[src], self.node_dict[dst]
        parent_node = self.node_dict[src_node.parent]
        assert (src_node.parent == dst_node.parent) and (src_node.h == dst_node.h)
        new_node = TreeNode(ID=self.node_number, partition=src_node.partition + dst_node.partition, pro_matrix=self.pro_matrix, h=src_node.h)
        self.node_dict[self.node_number] = new_node
        parent_node.del_children(src)
        parent_node.del_children(dst)
        parent_node.add_children(new_node.ID)
        new_node.add_children(src)
        new_node.add_children(dst)
        new_node.set_parent(parent_node.ID)
        src_node.set_parent(new_node.ID)
        dst_node.set_parent(new_node.ID)
        
        node_list = [src, dst]
        while len(node_list) > 0:
            old_node_list = deepcopy(node_list)
            node_list = list()
            for nid in old_node_list:
                self.node_dict[nid].h += 1
                if self.node_dict[nid].children is not None:
                    node_list += self.node_dict[nid].children
        self.node_number += 1
        return self.node_number - 1
    
    def get_depth(self, nid):
        depth = 0
        for pid in self.node_dict[nid].partition:
            depth = max(depth, self.node_dict[pid].h)
        return depth
    
    def optimize_tree(self, max_height=3):
        while True:
            max_se_red = 0.0
            src, dst, symbol  = -1, -1, -1
            for nid_0 in range(1, self.node_number):
                for nid_1 in range(nid_0 + 1, self.node_number):
                    node_0, node_1 = self.node_dict[nid_0], self.node_dict[nid_1]
                    if node_0 is None or node_1 is None:
                        continue
                    if (node_0.parent != node_1.parent) or (node_0.h != node_1.h):
                        continue
                    if (node_0.children is not None) and (node_1.children is not None):
                        se_red = self.cal_merge_ed(nid_0, nid_1)
                        if se_red > max_se_red:
                            max_se_red = se_red
                            src, dst = nid_0, nid_1
                            symbol = 0
                    if max(self.get_depth(nid_0), self.get_depth(nid_1)) < max_height:
                        se_red = self.cal_combine_ed(nid_0, nid_1)
                        if se_red > max_se_red:
                            max_se_red = se_red
                            src, dst = nid_0, nid_1
                            symbol = 1
            if max_se_red > 0.0:
                if symbol == 0:
                    self.exe_merge_op(src, dst)
                else:
                    self.exe_combine_op(src, dst)
            else:
                break
   
    def compute_transition_probability(self, src, dst):
        src_node, dst_node = self.node_dict[src], self.node_dict[dst]
        while dst not in src_node.partition:
            src_node = self.node_dict[src_node.parent]
        fenzi, fenmu = 0.0, 0.0
        while src_node.parent is not None:
            entropy = src_node.caculate_se(self.node_dict)
            fenzi += entropy
            fenmu += entropy
            src_node = self.node_dict[src_node.parent]
        while dst_node.ID != src_node.ID:
            fenmu += dst_node.caculate_se(self.node_dict)
            dst_node = self.node_dict[dst_node.parent]
        return fenzi / fenmu
    
    def adjust_paths(self, paths):
        for path in paths:
            for i in range(len(path) - 2):
                vid_src, vid_med, vid_dst = path[i], path[i + 1], path[i + 2]
                nid_src, nid_med, nid_dst = self.vertex_index_mapping[vid_src], self.vertex_index_mapping[vid_med], self.vertex_index_mapping[vid_dst]
                intersection = set(self.pro_graph.successors(vid_src)) & set(self.pro_graph.predecessors(vid_dst))
                old_pro = self.compute_transition_probability(nid_src, nid_med) + self.compute_transition_probability(nid_med, nid_dst)
                max_imp, new_vid = 0.0, 0
                for new_vid_med in intersection:
                    new_nid_med = self.vertex_index_mapping[new_vid_med]
                    imp = self.compute_transition_probability(nid_src, new_nid_med) + self.compute_transition_probability(new_nid_med, nid_dst) - old_pro
                    if imp > max_imp:
                        max_imp = imp
                        new_vid = new_vid_med
                if new_vid != 0:
                    path[i + 1] = new_vid
        return paths

    def obs_abstract(self):
        return self.stacked_features

    def get_community_representations(self):
        """
        为每个图节点检索其所处社区内具有最高分布概率的图节点特征表示，
        并构造与obs_abstract形状相同的输出
        
        返回:
            torch.Tensor: 形状与obs_abstract相同的张量，每个位置替换为对应社区的最高概率节点特征
        """
        # 获取每个社区的最高概率节点
        community_max_prob_nodes = {}
        
        # 遍历所有叶子节点(原始图节点)
        for nid in range(1, self.node_number):
            node = self.node_dict[nid]
            if node is None or node.children is not None:
                continue  # 跳过非叶子节点和已删除节点
            
            # 找到该节点所属的最高层社区(最小分区)
            current_node = node
            while True:
                parent_node = self.node_dict[current_node.parent]
                if parent_node is None or parent_node.ID == 0:  # 到达根节点
                    break
                current_node = parent_node
            
            # 当前节点代表一个社区，收集该社区所有原始节点
            community_nodes = []
            stack = [current_node]
            while stack:
                curr = stack.pop()
                if curr.children is None:  # 叶子节点(原始图节点)
                    community_nodes.append(curr.ID)
                elif curr.children is not None:
                    for child_id in curr.children:
                        if self.node_dict[child_id] is not None:
                            stack.append(self.node_dict[child_id])
            
            # 找出社区内概率最高的节点
            max_prob = -1
            max_prob_node_id = None
            for node_id in community_nodes:
                vertex_id = self.index_vertex_mapping[node_id]
                prob = nx.pagerank(self.pro_graph)[vertex_id]
                if prob > max_prob:
                    max_prob = prob
                    max_prob_node_id = node_id
            
            # 记录社区代表节点
            for node_id in community_nodes:
                community_max_prob_nodes[node_id] = max_prob_node_id
        
        # 重构特征张量
        new_features = []
        for path in self.paths:
            path_features = []
            for cid in path:
                node_id = self.vertex_index_mapping[cid]
                rep_node_id = community_max_prob_nodes[node_id]
                rep_cid = self.index_vertex_mapping[rep_node_id]
                path_features.append(self.rep_dict[rep_cid])
            new_features.append(torch.stack(path_features))
        
        return torch.stack(new_features)

import faiss
import torch.nn.functional as F
from tqdm import tqdm

class ObsSimilaritySearch:
    def __init__(self, train_loader, batch_size=128, device='cpu'):
        self.device = device
        self.index = None
        self.all_obs = None
        self._build_index(train_loader)
        self.batch_size = batch_size
    
    def _build_index(self, train_loader):
        sample_batch = next(iter(train_loader))
        self.obs_shape = sample_batch["obs"].shape[1:]  # 记录原始形状
        all_first_slices, all_obs = [], []
        for batch in tqdm(train_loader, desc="Loading first slices"):
            all_obs.append(batch["obs"].to(self.device))
            first_slices = batch["obs"][:, 0, :].to(self.device)
            all_first_slices.append(first_slices)
        self.all_obs = torch.cat(all_obs, dim=0)
        self.first_slices = torch.cat(all_first_slices, dim=0)

        self.index = faiss.IndexFlatIP(self.first_slices.shape[1])
        
        slices_np = self.first_slices.cpu().numpy().astype('float32')
        faiss.normalize_L2(slices_np)
        self.index.add(slices_np)
    
    def find_similar(self, current_obs, top_k=5):
        if len(current_obs.shape) == 3:
            current_obs = current_obs.squeeze(0)
        
        query_slice = current_obs[0].cpu().numpy().astype('float32')
        query_slice = query_slice / np.linalg.norm(query_slice)
        
        D, I = self.index.search(query_slice.reshape(1, -1), top_k)
        similar_obs = self.all_obs[torch.from_numpy(I[0]).to(self.device)]
        
        return {
            "obs": similar_obs,
            "first_slices": self.first_slices[I[0]],
            "indices": I[0],
            "similarities": D[0]
        }

    def get_original_batch(self, index):
        start_index= int(index / self.batch_size) * self.batch_size
        end_index = start_index + self.batch_size
        return self.all_obs[start_index: end_index]
