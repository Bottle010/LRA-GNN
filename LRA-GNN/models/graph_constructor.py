import torch
from torch_geometric.data import Data

def construct_initial_graph(keypoints, image_size=(224, 224), threshold=0.936):
    
    # Calculate Euclidean distances between keypoints
    num_keypoints = keypoints.size(0)
    keypoints = keypoints.float()
    distances = torch.cdist(keypoints, keypoints)  # Shape: (num_keypoints, num_keypoints)
    
    # Compute similarity matrix based on distances
    max_distance = torch.sqrt(torch.tensor(image_size[0]**2 + image_size[1]**2, dtype=torch.float32))
    similarity_matrix = 1 - distances / max_distance
    
    # Build edges based on the threshold
    edge_index = torch.nonzero(similarity_matrix > threshold, as_tuple=True)
    edge_index = torch.stack(edge_index, dim=0)
    
    # Initialize node features and edge features
    node_features = keypoints  # Use keypoints coordinates as node features
    edge_features = similarity_matrix[edge_index[0], edge_index[1]].unsqueeze(1)  # Use similarity as edge features
    
    # Create graph data object
    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    
    return graph

if __name__ == "__main__":

    # Assume a set of facial keypoints coordinates
    keypoints = torch.tensor([
        [50, 50],
        [100, 50],
        [75, 100],
        [125, 100]
    ], dtype=torch.float32)

    # Construct the initial graph
    graph = construct_initial_graph(keypoints)

    # Print the contents of the graph data object
    print("Node features (x):")
    print(graph.x)
    print("Edge indices (edge_index):")
    print(graph.edge_index)
    print("Edge features (edge_attr):")
    print(graph.edge_attr)