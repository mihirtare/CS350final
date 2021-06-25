# Author: Mihir Tare
# Created to fulfill the requirements for CS350: Data Science Third Year Project
from graphs import Graph
from simplicial_complex import SComplex
import numpy as np

# This is just an example interface for the objects.
# -------------------------------------------------
# Test Code For Simplicial Complex Object

# Example Simplices

# Complete Triangle: 2 dimensional complex
triangle_complex2d = [(0, 1, 2)]

# Complete Tetrahedron: 2 dimensional complex
tetrahedron_complex2d = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

# Complete Tetrahedron: 3 dimensional complex
tetrahedron_complex3d = [(0, 1, 2, 3)]

# Complete 5-Cell: 4 dimensional complex
five_cell_complex4d = [(0, 1, 2, 3, 4)]

# Fish Complex: 2 dimensional complex
fish_complex2d = [(0, 1, 2), (1, 2, 3), (3, 4, 5)]

# Fish Complex: 3 dimensional complex
fish_complex3d = [(0, 1, 2, 3), (3, 4, 5)]

# Diamond Complex: 2 dimensional complex
diamond_complex2d = [(0, 1, 2), (1, 2, 3)]

# Triangulated Torus: 2 dimensional complex
torus_complex2d = [(0, 1, 3), (0, 1, 6), (0, 4, 6), (0, 2, 7), (0, 3, 7), (0, 4, 2), (1, 2, 5), (1, 5, 3), (1, 2, 8), (1, 8, 6),
         (2, 7, 5), (2, 4, 8), (3, 5, 4), (3, 4, 8), (3, 7, 8), (4, 6, 5), (5, 6, 7), (6, 7, 8)]

# 3-regular Tree: 1 dimensional complex
regular_tree_complex1d = [(0, 1), (0, 9), (0, 15), (1, 2), (1, 3), (2, 4), (2, 9), (3, 5), (3, 15), (4, 7), (4, 6), (5, 8),
                (5, 6), (6, 12), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (7, 10), (8, 14), (7, 11),
                (8, 13)]


# Get numerical expansion criteria from all modes
def numerical_HDX_criteria(SC):
    gamma, lambda2, epsilon = SC.get_HDX_criteria()
    print("Random Walk gamma = " + str(gamma) + "\nLocal Expansion lambda2 = " + str(lambda2) + \
          "\nPercentage Convergence on Flattened Random Walk: " + str(epsilon) + "%")
    return gamma, lambda2, epsilon


def flattened_representation(SC):
    return SC.flattened_matrices()


def tensor_representation(SC):
    X = SC.tensor()
    print(X)
    return X


# Get a string summarizing the HDX status depending on chosen mode
# mode = {up_down, flattened, local}
def HDX_summary(SC, mode="up_down"):
    if mode == "flattened":
        print(SC.is_HDX(flattened=True))
    elif mode == "local":
        print(SC.is_HDX(local=True))
    else:
        print(SC.is_HDX(random_walk=True))
    return


# Plot the Simplicial Complex depending on chosen mode
# mode = {tree, flattened, underlying}
def plot_complex(SC, mode="tree"):
    if mode == "flattened":
        return SC.plot_flattened_complex()
    elif mode == "underlying":
        return SC.plot_underlying_link_graphs()
    else:
        return SC.plot_complex_tree()


def complex_distribution(SC):
    distribution = SC.get_distribution()
    for key in distribution.keys():
        print("\nDistribution over X(" + str(key-1) + ") is:")
        for simplex in distribution[key]:
            print("s = " + str(simplex) + ", pi_s = " + str(distribution[key][simplex]))
    return


# Generates k random complexes of dimension, d, and |V| = n
def simulate_random_complex(k=30, n=10, d=2):
    avg_g = 0
    avg_l = 0
    avg_e = 0
    for i in range(k):
        print("\nResults for Simulation #" + str(i+1) + ":")
        SC = SComplex(n=n, d=d)
        g, l, e = numerical_HDX_criteria(SC)
        avg_g += g
        avg_l += l
        avg_e += e
    print("\n\nSUMMARY of Random Walk Simulations on the Random Complex:")
    print("\nRandom Walk gamma = " + str(avg_g / k) + "\nLocal Expansion lambda2 = " + str(avg_l / k) + \
          "\nPercentage Convergence on Flattened Random Walk: " + str(avg_e / k) + "%")
    return avg_g/k, avg_l/k, avg_e/k


# Misc. method for sampling n random d-dimensional simplices from a complex using the distribution over X(d).
def sample_random_simplices(SC, n=10, d=1):
    distribution = SC.get_distribution()
    print("\nSampling " + str(n) + " random simplices from dimension "+ str(d) + " using the distribution:")
    edges = list(distribution[d+1].keys())
    indices = np.random.choice(np.arange(0, len(edges)), size=n, p=list(distribution[d+1].values()))
    random_edges = []
    for index in indices:
        random_edges.append(edges[index])
    return random_edges


'''
# Running the functions over a selected TestComplex
# simplices can be selected from the list of examples, or custom

simplices = tetrahedron_complex2d
TestComplex = SComplex(simplices)

numerical_HDX_criteria(TestComplex)
flattened_representation(TestComplex)
tensor_representation(TestComplex)

# Alternatively, mode="flattened" or mode="local" 
HDX_summary(TestComplex, mode="up_down")

# Alternatively, mode="flattened" or mode="underlying"
plot_complex(TestComplex, mode="tree")
complex_distribution(TestComplex)
simulate_random_complex()
random_simplices = sample_random_simplices(TestComplex), n=10, d=2)
print(random_simplices)
'''


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Test Code For Graph Object

# Example Graphs

regular_graph = {1: {(0,), (1,), (2,), (3,)},
                 2: {(0, 1), (0, 2), (1, 3), (2, 3)}}
irregular_graph = {1: {(0,), (1,), (2,), (3,)},
                   2: {(0, 2), (1, 3), (2, 3), (0, 3)}}
fish_graph = {1: [(0,), (1,), (2,), (3,), (4,), (5,)],
            2: [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)]}

'''
# Testing functions from Graph object 
# graph can be selected using the example graphs, or custom

test = irregular_graph
TestGraph = Graph(test)

print(TestGraph.adjacency())
title="irregular_graph"
TestGraph.display_graph(title=title)
print(TestGraph.get_expansion())
print(TestGraph.cheeger())
print(TestGraph.get_expansion(walk=True,criterion=1e-3))
'''