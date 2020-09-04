from collections import defaultdict
import sys
import numpy as np

print('setting recursion limit to 8000 in CALFIN/postprocessing/dfs.py')
sys.setrecursionlimit(8000)

def longest_DFS(edges, visited, index):
    """
        Description:
            Recursive algorithm to find the longest path from a specified node in an undirected weighted graph.
        Parameters:
            edges: (N, Kx) dictionary - list of edges and their respective end points/weights, per starting points.
            visited: (N) array like - list of nodes already visited in previous searches. Enforces path restriction of traversing each node once.
            index: integer - starting node index in edges.
    """
    visited[index] = 1
    max_path = [index]
    max_weight = -np.inf
    for edge in edges[index]:
        #Ensure neighbors have not laready been visited. 
        #f no neighbors are visited, current node index is a leaf, and default values are returned.
        if edge["end"] not in visited:
            #If path has not been traversed, do so, and cache it. Otherwise, use cached values.
            #Cache used in repeated dfs calls from longest_undirected_weighted_path
            #otherwise the brute force takes forever. cant believe this cache works
            if edge["max_path"] is None:
                edge["max_weight"], edge["max_path"] = longest_DFS(edges, visited, edge["end"])
            weight, path = edge["max_weight"], edge["max_path"]
            path_weight = edge["weight"] + weight
            #Store longest path
            if path_weight > max_weight:
                max_path = [index] + path
                max_weight = path_weight
    if np.isinf(max_weight):
        max_weight = 0
    return (max_weight, max_path)
    
def longest_undirected_weighted_path(graph):
    """
        Description:
            Algorithm to find the longest path in an undirected weighted graph.
            Performs 2 depth-first searches optimizing for length, to retrieve lognest overall path. Runs in O(N) time.
        Parameters:
            graph: (M, K) array_like - Symmetric adjacency/distance matrix of M vectors in K dimensions.
        Return Values:
            weight: int - length of longest path
            path: (P) array_like - list of indices correspodning to the longest path in the graph.
                
    """
    
    #Construct list of edges for each node
    edges = defaultdict(list)
    starts,ends = graph.nonzero()
    visited = defaultdict(int)
    for start,end in zip(starts,ends):
        edges[start].append({"end":end, "weight":graph[start,end], "max_path":None, "max_weight":None})

    #Must brute force to account for negative weighting and possibility of being trapped in local minima
    max_path = [starts[0]]
    max_weight = -np.inf
    #perform first DFS to find first endpoint of longest path,
    for i in range(len(starts)):
        visited = defaultdict(int)
        path_weight, path = longest_DFS(edges, visited, starts[i])
        if path_weight > max_weight:
            max_path = path
            max_weight = path_weight
    
#    #perform first DFS to find first endpoint of longest path,
#    visited = defaultdict(int)
#    weight, path = longest_DFS(edges, visited, starts[0])
#    
#    #perform second DFS to find second endpoint of longest path
#    visited = defaultdict(int)
#    endpoint_index = path[-1]
#    weight, path = longest_DFS(edges, visited, endpoint_index)
#    return weight, path
    return max_weight, max_path