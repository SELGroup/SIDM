from collections import defaultdict
import logging
from random import Random
import networkx as nx
import math
import numpy as np

class StructuralEntropyCalculator:
    def __init__(self, graph):
        self.G = graph
        self.G_debug=graph
        self.partition_list=[]
        self.partition = {}
        self.partition_size={}
        self.W = self.G.size(weight='weight')
        if self.W == 0:  # If the graph has no weights
            self.W = self.G.size()  # Consider each edge to have a weight of 1
        self.g_C = defaultdict(int)
        self.V_C = defaultdict(int)
        self.dlog2d_per_community = defaultdict(int)
        self.dlog2d_per_node = defaultdict(int)

    def set_partition(self, partition):
        self.partition = partition
        self._update_community_metrics()

    def _update_community_metrics(self):
        self.g_C.clear()
        self.V_C.clear()
        self.dlog2d_per_community.clear()
        for node, community_label in self.partition.items():
            self.partition_size[community_label]=self.partition_size.get(community_label,0)+1
            self.V_C[community_label] += self.G.degree(node, weight='weight') or 1
            for neighbor, data in self.G[node].items():
                if self.partition[neighbor] != community_label:
                    self.g_C[community_label] += data.get('weight', 1)
            self.dlog2d_per_node[node] = (self.G.degree(node, weight='weight') or 1
            ) * math.log2(self.G.degree(node, weight='weight') or 1)
            self.dlog2d_per_community[community_label] +=  self.dlog2d_per_node[node]

    def calculate_community_entropy(self, community_label):
        V_C = self.V_C[community_label]
        g_C = self.g_C[community_label]
        dlog2d_community = self.dlog2d_per_community[community_label]
        H_C = - (g_C / (2 * self.W)) * math.log2(V_C / (2 * self.W)) if V_C > 0 else 0
        H_C += (V_C / (2 * self.W)) * math.log2(V_C) if V_C > 0 else 0
        H_C -= dlog2d_community / (2 * self.W)
        return H_C

    def calculate_total_entropy(self):
        return sum(self.calculate_community_entropy(community_label) for community_label in set(self.partition.values()))

    def debug_partation_at_level(self,level):
        partition = self.partition_list[0].copy()
        for index in range(1, level + 1):
            for node, community in partition.items():
                partition[node] = self.partition_list[index][community]
        return partition
        
    def debug_calculate_entropy(self,infor=False):
        # Initialize variables
        G=self.G_debug
        partition=self.debug_partation_at_level(len(self.partition_list)-1) if len(self.partition_list)>0 else self.partition
        W = sum([G[u][v].get("weight", 1) for u, v in G.edges()])  # Total weight of edges in the graph
        communities = {}  # Community structure

        # Organize nodes into communities
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = set()
            if node in G:
                communities[comm_id].add(node)
            else:
                raise ValueError(f"Node {node} in partition is not in graph G.")

        # Calculate V_C for each community
        V_C = {comm: sum(G.degree(node, weight="weight") for node in nodes) for comm, nodes in communities.items()}

        # Calculate g_C for each community
        g_C = {}
        for comm, nodes in communities.items():
            edge_weight_sum = 0
            for node in nodes:
                for neighbor in G.neighbors(node):
                    if partition[neighbor] != comm:
                        edge_weight_sum += G[node][neighbor].get("weight", 1)
            g_C[comm] = edge_weight_sum

        # Calculate the first part of the entropy
        entropy = sum((-g_C[comm] / (2 * W)) * math.log2(V_C[comm] / (2 * W)) for comm in communities)

        # Calculate the second part of the entropy
        for node in G.nodes():
            d_v = G.degree(node, weight="weight")
            C_v = partition[node]
            entropy += (-d_v / (2 * W)) * math.log2(d_v / V_C[C_v])
        if infor:
            return {'g_C':dict(sorted(g_C.items())),'V_C':dict(sorted(V_C.items())),'partition':partition}
        return entropy

def print_cal(entropy_calculator):
    # Filter out zero values in g_C and V_C
    filtered_g_C = {k: v for k, v in entropy_calculator.g_C.items() if v != 0}
    filtered_V_C = {k: v for k, v in entropy_calculator.V_C.items() if v != 0}

    class_attributes_after_removal = {
        "g_C": filtered_g_C,
        "V_C": filtered_V_C,
        "partition": entropy_calculator.partition,
        "dlog2d_per_community": dict(entropy_calculator.dlog2d_per_community)
    }

    # Logging the filtered results
    logging.info(f"---{class_attributes_after_removal}")
    logging.info(f"---{entropy_calculator.calculate_total_entropy()}")  
    logging.info(f"---{entropy_calculator.debug_calculate_entropy(True)}")  
    logging.info(f"---{entropy_calculator.debug_calculate_entropy()}")

def remove_node(calculator, node):
    original_community = calculator.partition[node]
    original_community_entropy = calculator.calculate_community_entropy(original_community)
    # can be optimized, a flag like com_size==1
    #is_single_node_community = len([n for n in calculator.partition if calculator.partition[n] == original_community]) == 1
    is_single_node_community = calculator.partition_size[original_community] == 1
    if is_single_node_community:
        return 0
         
    else:
        # Assign a new community label to the removed node
        new_community_label = max(calculator.partition.values()) + 1
        calculator.partition[node] = new_community_label
        calculator.partition_size[new_community_label]=1
        calculator.partition_size[original_community]-=1
        # Update metrics for the original community, this seems right for supernode
        calculator.V_C[original_community] -= calculator.G.degree(node, weight='weight') or 1
        calculator.dlog2d_per_community[original_community] -= calculator.dlog2d_per_node[node]
        
        # Subtract the edges connecting the node to outside the original community
        for neighbor, data in calculator.G[node].items():
            if neighbor == node:
                continue
            if calculator.partition[neighbor] != original_community:
                calculator.g_C[original_community] -= data.get('weight', 1)
            else:
                calculator.g_C[original_community] += data.get('weight', 1)

        # Initialize metrics for the new community
        calculator.V_C[new_community_label] = calculator.G.degree(node, weight='weight') or 1
        calculator.dlog2d_per_community[new_community_label] = calculator.dlog2d_per_node[node]
        # Add the edges connecting the node to its former community
        # which is the degree of node- self loop*2
        calculator.g_C[new_community_label] = calculator.G.degree(node, weight='weight') - (calculator.G[node].get(node, {}).get('weight', 0) * 2)
        #calculator.g_C[new_community_label] = sum(calculator.G[node][neighbor].get('weight', 1) for neighbor in calculator.G[node] if calculator.partition[neighbor] != new_community_label)
    # Calculate new community entropy and total entropy change
    new_community_entropy = calculator.calculate_community_entropy(new_community_label)
    total_entropy_change = new_community_entropy+calculator.calculate_community_entropy(original_community) - original_community_entropy

    return total_entropy_change

def insert_node(calculator, node, com_label, try_insert=False): 
    temp_community_label = calculator.partition[node]
    if(temp_community_label == com_label):
        return 0
    original_community_entropy = calculator.calculate_community_entropy(com_label) +\
          calculator.calculate_community_entropy(temp_community_label)

    # Store the original state for the involved communities
    original_state = {
        'V_C_original': calculator.V_C.get(com_label, 0),
        'g_C_original': calculator.g_C.get(com_label, 0),
        'dlog2d_original': calculator.dlog2d_per_community.get(com_label, 0),
        'partition_size': calculator.partition_size.get(com_label, 0),  
        'V_C_temp': calculator.V_C.get(temp_community_label, 0),
        'g_C_temp': calculator.g_C.get(temp_community_label, 0),
        'dlog2d_temp': calculator.dlog2d_per_community.get(temp_community_label, 0),
        'partition_size_temp': calculator.partition_size.get(temp_community_label, 0),
        'partition_node': calculator.partition[node]
    }

    # Update partition
    calculator.partition[node] = com_label
    calculator.partition_size[com_label]+=1
    calculator.partition_size[temp_community_label]-=1  
    # Update metrics for the community
    calculator.V_C[com_label] += calculator.G.degree(node, weight='weight') or 1
    calculator.dlog2d_per_community[com_label] += calculator.dlog2d_per_node[node]
    for neighbor, data in calculator.G[node].items():
        if(neighbor == node):   
            continue
        if calculator.partition[neighbor] != temp_community_label:
            calculator.g_C[temp_community_label] -= data.get('weight', 1)
        else:
            calculator.g_C[temp_community_label] += data.get('weight', 1)
        
        if calculator.partition[neighbor] != com_label:
            calculator.g_C[com_label] += data.get('weight', 1)
        else:
            calculator.g_C[com_label] -= data.get('weight', 1)
    # Update metrics for the original community, when insert, original community is not necessary a singleton
    calculator.V_C[temp_community_label] -= calculator.G.degree(node, weight='weight') or 1
    calculator.dlog2d_per_community[temp_community_label] -= calculator.dlog2d_per_node[node]
    
    # Calculate new community entropy and total entropy change
    new_community_entropy = calculator.calculate_community_entropy(com_label)+\
        calculator.calculate_community_entropy(temp_community_label)
    total_entropy_change = new_community_entropy - original_community_entropy

    if try_insert:
        # Restore the original state for the involved communities
        calculator.V_C[com_label] = original_state['V_C_original']
        calculator.g_C[com_label] = original_state['g_C_original']
        calculator.dlog2d_per_community[com_label] = original_state['dlog2d_original']
        calculator.V_C[temp_community_label] = original_state['V_C_temp']
        calculator.g_C[temp_community_label] = original_state['g_C_temp']
        calculator.dlog2d_per_community[temp_community_label] = original_state['dlog2d_temp']
        calculator.partition[node] = original_state['partition_node']
    else:
        is_single_node_community = calculator.partition_size[temp_community_label] == 0
        #is_single_node_community = len([n for n in calculator.partition if calculator.partition[n] == temp_community_label]) == 0
      
        if is_single_node_community:
            # Delete the temporary community data from the calculator's attributes
            del calculator.g_C[temp_community_label]
            del calculator.V_C[temp_community_label]
            del calculator.dlog2d_per_community[temp_community_label]
            del calculator.dlog2d_per_node[node]  
            del calculator.partition_size[temp_community_label]   
    return total_entropy_change

# Remove node 'A' and then insert it back into its original community
def induced_graph(partition, graph):
    """Produce the graph where nodes are the communities.

    There is a link of weight w between communities if the sum of the weights
    of the links between their elements is w.

    Parameters
    ----------
    partition : dict
       A dictionary where keys are graph nodes and values the part the node
       belongs to.
    graph : networkx.Graph
        The initial graph.

    Returns
    -------
    g : networkx.Graph
       A networkx graph where nodes are the parts.
    """
    induced_g = nx.Graph()
    # Initialize weight between communities
    for community in set(partition.values()):
        induced_g.add_node(community)
    # Aggregate weight of edges between communities
    for (node1, node2, data) in graph.edges(data=True):
        com1 = partition[node1]
        com2 = partition[node2]
        weight = data.get('weight', 1)
        if induced_g.has_edge(com1, com2):
            induced_g[com1][com2]['weight'] += weight
        else:
            induced_g.add_edge(com1, com2, weight=weight)
    return induced_g

def induced_calculator(partition, induced_graph, entropy_calculator):
    # Update the graph in the entropy calculator
    entropy_calculator.G = induced_graph

    # Update the partition - each node (community) in the induced graph is its own community
    new_partition = {node: node for node in induced_graph.nodes()}
    # only attribute that needs to be updated is the dlogd and partition size
    entropy_calculator.partition=new_partition
    for node,partition in entropy_calculator.partition.items():
        entropy_calculator.dlog2d_per_node[node] = entropy_calculator.dlog2d_per_community[partition]
        entropy_calculator.partition_size[partition]=1
    return entropy_calculator

__PASS_MAX=-1
__MIN = 0.0000001



def __randomize(items, random_state):
    """Returns a List containing a random permutation of items"""
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items

def __one_level(graph, entropy_calculator , random_state): 
    """
    Compute one level of communities

    This function iteratively goes through each node in the graph, attempting to move the node to a different
    community to maximize modularity. A 'level' consists of a complete pass through all nodes. The function stops
    either when no further improvement in modularity is possible or when a maximum number of passes (__PASS_MAX)
    is reached.

    Parameters:
    - graph (nx.Graph): The graph over which communities are being optimized.
    - entropy_calculator (StructuralEntropyCalculatorUpdated): An object maintaining the state of the graph, including the current community
                       assignment of nodes and other relevant properties for entropy calculation
    - random_state (np.random.RandomState): A RandomState instance for reproducible randomness.

    Returns:
    - None: The function updates the 'status' object in-place, recording community assignments.
    """
    modified = True
    nb_pass_done = 0
    cur_entropy = entropy_calculator.calculate_total_entropy()
    new_entropy = cur_entropy

    while modified and nb_pass_done != __PASS_MAX:
        cur_entropy = new_entropy
        modified = False
        nb_pass_done += 1

        for node in __randomize(graph.nodes(), random_state):
            com_node = entropy_calculator.partition[node]
            if(node=='L2_3_L'):
                x=1
            remove_cost = remove_node(entropy_calculator,node)
            
            best_com = com_node
            best_decrease = 1e20
            
            node_and_neighbors = [node] + list(entropy_calculator.G[node])
            for neighbor_or_self in __randomize(node_and_neighbors, random_state):
                com=entropy_calculator.partition[neighbor_or_self]
                dec = remove_cost + insert_node(entropy_calculator,node, com,try_insert=True)
                logging.debug(f"if move Node: {node}, to Neighbor/Self: {neighbor_or_self}, in Community: {com}, will Change: {dec}")
                if dec < best_decrease:
                    best_decrease = dec
                    best_com = com
            #L3_3_U:26, this node is put into com 25 before, which meas com 26 is deleted
            # but later L2_3_L is put into that community
            # actually, it's the new added community when removing
            if(node=='L3_3_U'):
                x=1
            if(node=='L2_3_L'):
                x=1
            if(best_com==26):
                x=1
            if(node=='L1_1_L'):
                x=1
            insert_node(entropy_calculator,node, best_com)
            logging.info(f"inserting Node: {node} to com: {best_com} ")
            if best_com != com_node:
                modified = True

        new_entropy = entropy_calculator.calculate_total_entropy()
        if new_entropy - cur_entropy < __MIN:
            logging.info(f"one level done because of no improvement, with entropy {new_entropy} ")
            print_cal(entropy_calculator)
            break

def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    """
    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret
def generate_dendrogram(graph,
                        part_init=None,
                        random_state=None):
    """Find communities in the graph and return the associated dendrogram

    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1. The higher the level is, the bigger
    are the communities


    Parameters
    ----------
    graph : networkx.Graph
        the networkx graph which will be decomposed

    Returns
    -------
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph

    Raises
    ------
    TypeError
        If the graph is not a networkx.Graph

    See Also
    --------
    best_partition

    Notes
    -----
    Uses Louvain algorithm

    References
    ----------
    .. 1. Blondel, V.D. et al. Fast unfolding of communities in large
    networks. J. Stat. Mech 10008, 1-12(2008).

    Examples
    --------
    >>> G=nx.erdos_renyi_graph(100, 0.01)
    >>> dendo = generate_dendrogram(G)
    >>> for level in range(len(dendo) - 1) :
    >>>     print("partition at level", level,
    >>>           "is", partition_at_level(dendo, level))
    :param weight:
    :type weight:
    """
    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()
    if part_init==None:
        part_init = {node: i for i, node in enumerate(graph.nodes())}
        #logging.info(f"initial partition: {part_init}")
    if random_state==None:
        random_state=Random(42)
    entropy_calculator = StructuralEntropyCalculator(graph)
    entropy_calculator.set_partition(part_init) 
    status_list = list()
    __one_level(current_graph, entropy_calculator, random_state)
    new_entropy = entropy_calculator.calculate_total_entropy()
    entropy_calculator.partition_list.append(entropy_calculator.partition)
    status_list.append(entropy_calculator.partition)
    entropy = new_entropy
    current_graph = induced_graph(entropy_calculator.partition, current_graph)
    entropy_calculator= induced_calculator(entropy_calculator.partition,current_graph,entropy_calculator)
    logging.info("first one level result:")
    print_cal(entropy_calculator)
    while True:
        #print("------------------>")
        __one_level(current_graph, entropy_calculator, random_state)
        new_entropy = entropy_calculator.calculate_total_entropy()
        if abs(new_entropy - entropy) < __MIN:
            break
        status_list.append(entropy_calculator.partition)
        entropy_calculator.partition_list.append(entropy_calculator.partition)
        entropy = new_entropy
        current_graph = induced_graph(entropy_calculator.partition, current_graph)
        entropy_calculator= induced_calculator(entropy_calculator.partition,current_graph,entropy_calculator)
        logging.info("Level completed. entropy: {}".format(new_entropy))
        logging.info("Current community assignments: {}".format(entropy_calculator.partition))
        print_cal(entropy_calculator)
    #print(status_list)
    return status_list[:]

def partition_at_level(dendrogram, level):
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition

def best_partition(graph,
                   partition=None,
                   random_state=None):
    dendo = generate_dendrogram(graph,
                                partition,
                                random_state)
    return partition_at_level(dendo, len(dendo) - 1)

# from random import Random
# def test():
#     random_state = Random(42)
#     G = nx.Graph()
#     edges = [("A", "B"), ("B", "C"), ("C", "D")]
#     G.add_edges_from(edges)
#     entropy_calculator = StructuralEntropyCalculator(G)
#     initial_partition = {"A": 1, "B": 2, "C": 3, "D": 4}
#     #initial_partition = {"A": 1, "B": 1, "C": 2, "D": 2}
#     entropy_calculator.set_partition(initial_partition)

#     initial_entropy = entropy_calculator.calculate_total_entropy()
#     generate_dendrogram(G,initial_partition,random_state)

# def test_karate():
#     # Set random state
#     random_state = Random(42)

#     # Create a Karate Graph
#     G = nx.karate_club_graph()

#     # Initialize the entropy calculator with the graph
#     entropy_calculator = StructuralEntropyCalculator(G)

#     # Define initial partitions where each node is in its own partition
#     initial_partition = {node: i for i, node in enumerate(G.nodes())}

#     # Set the partition in the entropy calculator
#     entropy_calculator.set_partition(initial_partition)

#     # Calculate the initial entropy
#     initial_entropy = entropy_calculator.calculate_total_entropy()

#     # Generate a dendrogram based on the graph and initial partition
#     generate_dendrogram(G, initial_partition)
#     result=best_partition(G)
#     print(result)



def test_triangle(lv):
    import networkx as nx
    import matplotlib.pyplot as plt
    import math

    def calculate_triangle_vertices(center, edge_length):
        """
        Calculate the vertices of an equilateral triangle given a center and edge length.
        """
        height = (math.sqrt(3) / 2) * edge_length
        vertex1 = (center[0] - edge_length / 2, center[1] - height / 3)
        vertex2 = (center[0] + edge_length / 2, center[1] - height / 3)
        vertex3 = (center[0], center[1] + 2 * height / 3)
        return [vertex1, vertex2, vertex3]

    def generate_hierarchical_triangle(G, level, center, edge_length, name_prefix):
        """
        Recursively generates a hierarchical triangle structure with corrected and robust connections.
        """
        if level == 1:
            # Base case: create a level 1 triangle
            vertices = calculate_triangle_vertices(center, edge_length)
            G.add_node(f'{name_prefix}L', pos=vertices[0])  # Left node
            G.add_node(f'{name_prefix}R', pos=vertices[1])  # Right node
            G.add_node(f'{name_prefix}U', pos=vertices[2])  # Up node
            G.add_edges_from([(f'{name_prefix}L', f'{name_prefix}R'), 
                            (f'{name_prefix}R', f'{name_prefix}U'), 
                            (f'{name_prefix}U', f'{name_prefix}L')])
            return [f'{name_prefix}L', f'{name_prefix}R', f'{name_prefix}U']
        else:
            # Recursive case: generate 3 smaller triangles and connect them
            sub_edge_length = edge_length / 2  # Adjusted for geometric progression
            sub_triangle_centers = calculate_triangle_vertices(center, edge_length)

            # Generate and store nodes for each sub-triangle
            sub_triangles_nodes = []
            for i, sub_center in enumerate(sub_triangle_centers):
                sub_nodes = generate_hierarchical_triangle(G, level - 1, sub_center, sub_edge_length, f'{name_prefix}{i+1}_')
                sub_triangles_nodes.append(sub_nodes)

            # Connect adjacent triangles according to new rules
            # Connect right node of left triangle to left node of right triangle
            G.add_edge(sub_triangles_nodes[0][1], sub_triangles_nodes[1][0])
            # Connect left node of up triangle to up node of left triangle
            G.add_edge(sub_triangles_nodes[2][0], sub_triangles_nodes[0][2])
            # Connect right node of up triangle to up node of right triangle
            G.add_edge(sub_triangles_nodes[2][1], sub_triangles_nodes[1][2])

            return [sub_triangles_nodes[0][0], sub_triangles_nodes[1][1], sub_triangles_nodes[2][2]]

    # Create the graph
    G = nx.Graph()
    level = lv # Testing with level 3
    center = (0, 0)  # Center of the top-level triangle
    edge_length = 1  # Edge length for the top-level triangle
    generate_hierarchical_triangle(G, level, center, edge_length, 'L')

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title(f"Hierarchical Triangle Structure (Level {level})")
    plt.show()
    plt.savefig('my_plot.png')

    generate_dendrogram(G)
    result=best_partition(G)
    print(result)
    # Assign a color to each community
    unique_communities = set(result.values())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_to_color = {community: color for community, color in zip(unique_communities, colors)}

    # Create a list of colors for each node
    node_colors = [community_to_color[result[node]] for node in G.nodes()]

    # Draw the graph with community colors
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
    plt.title(f"Hierarchical Triangle Structure with Communities (Level {level})")
    plt.show()
    plt.savefig('community_plot.png')


import networkx as nx
from cdlib import NodeClustering

import networkx as nx
from cdlib import NodeClustering

def SILouvain(G: nx.Graph) -> NodeClustering:
    """
    Run the best_partition to get a vertex:community dictionary.
    Convert the dictionary to a NodeClustering object.
    Communities is a list of community member lists.
    # Example usage
    # G = nx.Graph()  # or any networkx graph
    # clustering = SILouvain(G)
    """
    # Run best_partition to get the node-community mapping
    partition = best_partition(G)

    # Convert the partition into a list of communities
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)
    
    # Convert the dictionary to a list of lists
    communities = list(communities.values())

    # Create and return the NodeClustering object
    return NodeClustering(communities=communities, graph=G, method_name='SILouvain', method_parameters={}, overlap=False)

from scipy.spatial.distance import pdist, squareform

def si_abstract(data):
    rs_data = data.reshape([-1, data.shape[-1]]).cpu().detach().numpy()  # [batch_size, seq_len, data_dimmension] -> [batch_size * seq_len, data_dimmension]
    # 计算行向量间的余弦相似度
    cosine_similarity = 1 - squareform(pdist(rs_data, 'cosine'))
    # 对于cosine_similarity中大于0.3的元素保留其值，其他设置为0
    filtered_similarity = np.where(cosine_similarity > 0.3, cosine_similarity, 0)
    # 归一化处理：确保所有元素之和为1
    sum_of_elements = np.sum(filtered_similarity)
    if sum_of_elements != 0:  # 防止除以零
        normalized_adjacency_matrix = filtered_similarity / sum_of_elements
    else:
        normalized_adjacency_matrix = filtered_similarity
    # 使用处理后的邻接矩阵创建 NetworkX 图
    G = nx.from_numpy_matrix(normalized_adjacency_matrix)
    clustering = SILouvain(G).communities
    print(clustering)
