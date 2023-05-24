import numpy as np
import torch as th
import random
from SEP import PartitionTree,calculate_adj_matrix

def create_sample():
    # For now, let's create a random tensor to simulate the input to your function
    sample = th.randn(100, 10)
    sample_np = sample.numpy()
    return sample_np

def test_calculate_adj_matrix():
    sample_np = create_sample()

    # Create a similarity graph
    similarity_graph = calculate_adj_matrix(sample_np)
    print(similarity_graph)

def test_partition_tree():
    sample_np = create_sample()

    # Create a similarity graph
    similarity_graph = calculate_adj_matrix(sample_np)

    # Perform custom clustering using the partition tree
    y = PartitionTree(adj_matrix=similarity_graph)
    y.build_coding_tree(2)

def test_centroids_calculation():
    sample_np = create_sample()

    # Create a similarity graph
    similarity_graph = calculate_adj_matrix(sample_np)

    # Perform custom clustering using the partition tree
    y = PartitionTree(adj_matrix=similarity_graph)
    y.build_coding_tree(2)

    # Compute cluster centroids
    centroids = []
    for node in y.tree_node.values():
        if node.children is not None:  # ignore leaf nodes
            try:
                cluster_points = sample_np[node.partition]
                centroids.append(np.mean(cluster_points, axis=0))
            except IndexError as e:
                print("Error: ", e)
                print("node.partition: ", node.partition)
                print("Size of sample_np: ", len(sample_np))
                valid_partition_indices = [i for i in node.partition if i < len(sample_np)]
                if valid_partition_indices:  # proceed only if there are valid indices
                    cluster_points = sample_np[valid_partition_indices]
                    centroids.append(np.mean(cluster_points, axis=0))

    print("Centroids:", centroids)

def main():
    test_calculate_adj_matrix()
    test_partition_tree()
    test_centroids_calculation()

if __name__ == "__main__":
    main()
