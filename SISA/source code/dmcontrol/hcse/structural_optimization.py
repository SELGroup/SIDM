#!/usr/bin/env python
# encoding: utf-8
# author:  ryan_wu
# email:   imitator_wu@outlook.com
# date:    2020-11-26 16:09:29
# 把networkx的图转成邻接矩阵,然后用结构信息树，给转成各种深度的树
####### 各树节点的 vol 计算 #######

import os
from readline import set_completion_display_matches_hook
import sys
import copy
import json
import time
import pickle
import itertools
import traceback
import numpy as np
import networkx as nx
import pandas as pd
import math
import torch
# from multiprocessing import Pool
from hcse.lib.encoding_tree import PartitionTree, PartitionTreeNode


PWD = os.path.dirname(os.path.realpath(__file__))


def trans_to_adj(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    nodes = range(len(graph.nodes))
    return nx.to_numpy_array(graph, nodelist=nodes)

def trans_to_tree(adj, k=2):
    undirected_adj = np.array(adj)
    y = PartitionTree(adj_matrix=undirected_adj)
    x = y.build_encoding_tree(k)
    return y.tree_node


def update_depth(tree):
    # set leaf depth
    wait_update = [k for k, v in tree.items() if v.children is None]
    while wait_update:
        for nid in wait_update:
            node = tree[nid]
            if node.children is None:
                node.child_h = 0
            else:
                node.child_h = tree[list(node.children)[0]].child_h + 1
        wait_update = set([tree[nid].parent for nid in wait_update if tree[nid].parent])

def update_node(tree):
    update_depth(tree)
    d_id= [(v.child_h, v.ID) for k, v in tree.items()]
    d_id.sort()
    new_tree = {}
    for k, v in tree.items():
        n = copy.deepcopy(v)
        n.ID = d_id.index((n.child_h, n.ID))
        if n.parent is not None:
            n.parent = d_id.index((n.child_h+1, n.parent))
        if n.children is not None:
            n.children = [d_id.index((n.child_h-1, c)) for c in n.children]
        n = n.__dict__
        n['depth'] = n['child_h']
        new_tree[n['ID']] = n
    return new_tree

def pool_trans(graph, tree_depth):
    adj_mat = trans_to_adj(graph)
    tree = trans_to_tree(adj_mat, tree_depth)
    tree = update_node(tree)
    return tree

def pool_trans_disconnected(graph, tree_depth):
    # 一个数据集里也有连通的图
    if nx.is_connected(graph):
        return pool_trans(graph, tree_depth)
    trees = []
    # gi: 用来标记是第几个子图, graph index
    for gi, sub_nodes in enumerate(nx.connected_components(graph)):
        if len(sub_nodes) == 1:
        # 编码树没办法处理单个点, 手动组成单节点树
            node = list(sub_nodes)[0]
            # leaf node, parent的组成：graphIndex_layerIndex_nodeIndex
            js = [{'ID': node, 'parent': '%s_%s_0' % (gi, 1), 'depth': 0, 'children': None}]
            for d in range(1, tree_depth+1):
                js.append({'ID': '%s_%s_0' % (gi, d),
                           'parent': '%s_%s_0' % (gi, d+1) if d<tree_depth else None,
                           'depth': d,
                           'children': [js[-1]['ID']]
                          })
        else:
            sg = graph.subgraph(sub_nodes) # sub graph
            nodes = list(sg.nodes)
            nodes.sort()
            nmap = {n: nodes.index(n) for n in nodes}
            sg = nx.relabel_nodes(sg, nmap)
            # 图转树
            adj_mat = trans_to_adj(sg)
            tree = trans_to_tree(adj_mat, tree_depth)
            tree = update_node(tree)
            # relable tree id
            js = list(tree.values())
            rmap = {nodes.index(n): n for n in nodes}  # 叶子节点用原ID
            for j in js:
                if j['depth'] > 0:
                    rmap[j['ID']] = '%s_%s_%s' % (gi, j['depth'], j['ID'])
            for j in js:
                j['ID'] = rmap[j['ID']]
                j['parent'] = rmap[j['parent']] if j['depth']<tree_depth else None
                j['children'] = [rmap[c] for c in j['children']] if j['children'] else None
        trees.append(js)
    # 整树节点id relabel
    id_map = {}
    for d in range(0, tree_depth+1):
        for js in trees:
            for j in js:
                if j['depth'] == d:
                    # 叶子节点维持原图ID
                    id_map[j['ID']] = len(id_map) if d>0 else j['ID']
    tree = {}
    root_ids = []
    for js in trees:
        for j in js:
            n = copy.deepcopy(j)
            n['parent'] = id_map[n['parent']] if n['parent'] else None
            n['children'] = [id_map[c] for c in n['children']] if n['children'] else None
            n['ID'] = id_map[n['ID']]
            tree[n['ID']] = n
            if n['parent'] is None:
                root_ids.append(n['ID'])
    # 根节点合并
    root_id = min(root_ids)
    root_children = list(itertools.chain.from_iterable([tree[i]['children'] for i in root_ids]))
    root_node = {'ID': root_id, 'parent': None, 'children': root_children, 'depth': tree_depth}
    [tree.pop(i) for i in root_ids] # 删掉所有根节点
    for c in root_children: # 修改中间节点到根节点的映射到最新根节点
        tree[c]['parent'] = root_id
    tree[root_id] = root_node # 加入根节
    return tree

def caculate_1d_se(edges, node_number):
    degree = []
    for _ in range(node_number):
        degree.append(0.0)
    for edge in edges:
        degree[edge[0]] += edge[2]
        degree[edge[1]] += edge[2]
    vol = 0.0
    for value in degree:
        vol += value
    se = 0.0
    for value in degree:
        if value > 0.0:
            se += - value / vol * math.log2(value / vol)
    return se

def filter_edge(edges, node_number):
    ses = []
    pds = []
    for k in range(1, node_number):
        tws = []
        for i in range(node_number):
            ws = []
            for edge in edges:
                if edge[0] == i or edge[1] == i:
                    ws.append(edge[2])
            ws.sort()
            tws.append(ws[-k])
        filtered_edges = []
        for edge in edges:
            if edge[2] >= tws[edge[0]] and edge[2] >= tws[edge[1]]:
                filtered_edges.append([edge[0], edge[1], edge[2]])
        ses.append(caculate_1d_se(filtered_edges, node_number))
        pds.append(filtered_edges)
        if k > 2:
            if ses[k - 2] < ses[k - 3] and ses[k - 2] < ses[k - 1]:
                break
    print(k - 2)
    return pds[k - 2]

def caculate_se(tree, nid, vg):
    if tree[nid]['parent'] is None or 'g' not in tree[nid].keys() or 'vol' not in tree[nid].keys():
        return 0.0
    gi = tree[nid]['g']
    vi = tree[nid]['vol']
    if 'vol' not in tree[tree[nid]['parent']].keys():
        vp = vg
    else:
        vp = tree[tree[nid]['parent']]['vol']
    se = - gi / vg * math.log(vi / vp, 2)
    return se

def get_two_dimensional_partitions(tree):
    rid = 0
    for vid in tree.keys():
        if tree[vid]['parent'] is None:
            rid = vid
            break
    partitions = dict()
    for label, cid in enumerate(tree[rid]['children']):
        partitions[label] = tree[cid]['children']
    return partitions, tree[rid]['children']

def aggregate_by_id(tree, vid, vg, z):
    if tree[vid]['children'] is None:
        return z[vid]
    ps, zs = [], []
    p_sum = 0.0
    for cid in tree[vid]['children']:
        p = math.exp(-caculate_se(tree, cid, vg))
        ps.append(p)
        zs.append(aggregate_by_id(tree, cid, vg, z))
        p_sum += p
    az_vid = (ps[0]/ p_sum) * zs[0]
    for index in range(1, len(ps)):
        az_vid += (ps[index] / p_sum) * zs[index]
    return az_vid

def get_partition_of_state(similarity_matrix, z):
    edges = []
    node_num = similarity_matrix.shape[0]
    for nid1 in range(node_num):
        for nid2 in range(node_num):
            if nid1 >= nid2:
                continue
            edges.append([nid1, nid2, similarity_matrix[nid1, nid2]])
    filtered_edges = filter_edge(edges, node_num)
    vg = 0.0
    for edge in filtered_edges:
        vg += 2 * edge[2]
    graph = nx.Graph()
    for nid in range(node_num):
        graph.add_node(nid)
    graph.add_weighted_edges_from(filtered_edges)
    tree = pool_trans_disconnected(graph, 2)
    partitions, children = get_two_dimensional_partitions(tree)
    azs = aggregate_by_id(tree, children[0], vg, z).unsqueeze(0)
    for index, cid in enumerate(children):
        if index == 0:
            continue
        azs = torch.cat((azs, aggregate_by_id(tree, cid, vg, z).unsqueeze(0)), dim=0)
    return partitions, azs

def get_descendant_by_id(tree, vid, descendants):
    if tree[vid]['children'] is None:
        descendants[vid] = [vid]
        return descendants
    ds = []
    for cid in tree[vid]['children']:
        descendants = get_descendant_by_id(tree, cid, descendants)
    for cid in tree[vid]['children']:
        for nid in descendants[cid]:
            ds.append(nid)
    descendants[vid] = ds
    return descendants

def caculate_strutural_probability(tree, nodes, vg):
    sp_sum = 0.0
    for nid in nodes:
        vid = nid
        se = 0.0
        while tree[vid]['parent'] is not None:
            se += caculate_se(tree, vid, vg)
            vid = tree[vid]['parent']
        sp_sum += math.exp(-se)
    return sp_sum

def caculate_boundary_probability(tree, vid, vg, sp_sum):
    se = 0.0
    while tree[vid]['parent'] is not None:
        se += caculate_se(tree, vid, vg)
        vid = tree[vid]['parent']
    return math.exp(-se) / sp_sum

def caculate_condifitional_se(tree, nid1, nid2, descendants, vg):
    vid1, vid2 = nid1, nid2
    while nid2 not in descendants[vid1]:
        vid1 = tree[vid1]['parent']
    se = 0.0
    while vid2 != vid1:
        se += caculate_se(tree, vid2, vg)
        vid2 = tree[vid2]['parent']
    return se

def construct_graph(nodes, edges):
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)
    vg = 0.0
    for edge in edges:
        vg += edge[2]
    graph.add_weighted_edges_from(edges)
    tree = pool_trans_disconnected(graph, 2)
    sp_sum = caculate_strutural_probability(tree, nodes, vg)
    bp_vector = []
    for node in nodes:
        bp_vector.append(caculate_boundary_probability(tree, node, vg, sp_sum))
    rid = 0
    for vid in tree.keys():
        if tree[vid]['parent'] is None:
            rid = vid
            break
    descendants = dict()
    descendants = get_descendant_by_id(tree, rid, descendants)
    cp_matrix = []
    for nid1 in nodes:
        cp_vector = []
        for nid2 in nodes:
            if nid1 == nid2:
                cp_vector.append(0.0)
                continue
            cp_vector.append(caculate_condifitional_se(tree, nid1, nid2, descendants, vg))
        cp_matrix.append(cp_vector)
    cp_sum = []
    for nid in nodes:
        cp_sum.append(sum(cp_matrix[nid]))
    for nid1 in nodes:
        for nid2 in nodes:
            cp_matrix[nid1][nid2] /= cp_sum[nid1]
    return bp_vector, cp_matrix
