# Author: Mihir Tare
# Created to fulfill the requirements for CS350: Data Science Third Year Project
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
from graphs import Graph
import math
import random


class SComplex(object):

    def __init__(self, simplices=[], d=0, n=0):
        # If Simplices not specified, the object initialises to a complete 3-dimensional tetrahedron complex
        if n > 0:
            simplices = self.__random_complex(n, d)
        elif len(simplices) == 0:
            simplices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        self.__simplices = simplices

    @staticmethod
    def __random_complex(n, d):
        simplices = []
        vertices = []
        for j in range(n):
            simplices.append((j,))
            vertices.append(j)
        for i in range(1, d + 1):
            comb = iter.combinations(vertices, i + 1)
            for item in comb:
                if np.random.random_sample() > 0.5:
                    simplices.append(item)
        return simplices

    def vertices(self):
        unique_elements = set()
        for simplex in self.simplices():
            # Adding elements to a set which prevents duplicates
            for elem in simplex:
                unique_elements.add(elem)
        return unique_elements

    def vertex_count(self):
        return len(self.vertices())

    def simplices(self):
        return self.__simplices

    def simplex_count(self):
        return len(self.simplices())

    def simplex_dictionary(self):
        # Creating a partitioning for the Simplicial Complex
        __simplex_dict = self.__get_simplex_dict()
        ret_str = ""
        for key in __simplex_dict.keys():
            ret_str += "\nX(" + str(key - 1) + "):" + str(__simplex_dict[key])

        return ret_str

    def tensor_dictionary(self):
        # Retrieve the full dictionary storing the simplicial complex for displaying.
        __simplex_dict = self.__get_simplex_dict()
        __simplex_dict = self.__convert_simplices(__simplex_dict)
        ret_str = ""
        for key in __simplex_dict.keys():
            ret_str += "Edges in dimension " + str(key - 1) + " are indexed as:" + str(__simplex_dict[key]) + \
                       str(len(__simplex_dict[key])) + "\n"
        return ret_str

    def flattened_dictionary(self):
        return self.__get_flattened_dict()

    def flattened_matrices(self):
        # Find and display the adjacency matrices of all flattened matrices along with a guide explaining the
        # indexing to the matrices.
        fm = self.__get_flattened_dict()
        __vertex_dict = self.__get_vertex_dict()

        for key in self.__get_keys():
            # print("The number of edges in dimension ", key, " are: ", index)
            graph = fm[key]
            print("\n** VERTEX GUIDE FOR DIMENSION", key - 1, "** \n(simplex): matrix index")
            for item in __vertex_dict[key]:
                print(item, ": ", __vertex_dict[key][item])
            print("The simplices in dimension", key - 1, "are shown by the matrix:\n", graph.adjacency())

        return self.plot_flattened_complex()

    def tensor(self):
        return self.__get_tensor()

    def links(self, vertex):
        ret_str = "The link for the simplex, S = {" + str(vertex) + "} is: \nXs = " + str(self.__get_link(vertex))
        return ret_str

    def underlying_link_graphs(self):
        # Return a string containing the underlying graphs of all links and their adjacency matrices.
        vertices = self.vertices()
        ret_str = ""
        for vertex in vertices:
            link = self.__get_link((vertex,))
            if len(link) > 0:
                underlying_graph_dict = self.__get_underlying_graph(link)
                underlying_graph = Graph(underlying_graph_dict)
                adjacency_matrix = underlying_graph.normalised_adjacency()
                ret_str += "The link for the simplex, S = {" + str(vertex) + "} is: \nXs = " + str(link) + \
                           "\nThe link has an underlying graph with vertices and edges:, \n" + str(underlying_graph) + \
                           "\nThe underlying graph has an adjacency matrix, G = \n" + str(adjacency_matrix) + "\n\n"
            else:
                ret_str += "The simplex, S = {" + str(vertex) + "} does not have a link and has no underlying graph." \
                           + "\n\n"
        return ret_str

    def is_HDX(self, local=False, flattened=False, random_walk=False):
        # Get the HDX status based on which modes are chosen
        ret_str = ""
        if flattened:
            ret_str += self.__get_flattened_HDX()
            ret_str += "\n"
        if local:
            ret_str += self.__get_local_HDX_status()
            ret_str += "\n"
        if random_walk or len(ret_str) == 0:
            ret_str += self.__get_complex_random_walk_HDX()

        print(ret_str)

    def get_HDX_criteria(self, local=True, flattened=True, random_walk=True):
        # Returns numerical values of the expansion criteria in the order:
        # up-down random walk expansion, local expansion and flattened random walk expansion
        # Flattened expansion is calculated in terms of % of times the random walk converges to
        # its stationary distribution.
        if flattened:
            epsilon = self.__get_epsilon()
        if local:
            lambda2 = self.__get_lambda()
        if random_walk:
            gamma = self.__get_gamma()
        return gamma, lambda2, epsilon * 100

    def plot_underlying_link_graphs(self):
        return self.__plot_underlying_graphs()

    def plot_flattened_complex(self):
        return self.__get_flattened_graphs()

    def plot_complex_tree(self):
        return self.__plot_complex_tree()

    def get_distribution(self):
        return self.__get_weighted_distribution()[0]

    def __get_lambda(self):
        base_dict = self.__get_underlying_graph(self.simplices())
        base_graph = Graph(base_dict)
        max_expansion = 1e-999
        expansion = base_graph.get_expansion()
        if type(expansion) is not str:
            if expansion > max_expansion: max_expansion = expansion

        __simplex_dict = self.__get_simplex_dict()
        max_dim = self.__get_simplex_dim()

        for key in __simplex_dict.keys():
            if key > max_dim - 2:
                continue
            for simplex in __simplex_dict[key]:
                link = self.__get_link(simplex)
                if len(link) > 0:
                    underlying_graph_dict = self.__get_underlying_graph(link)
                    underlying_graph = Graph(underlying_graph_dict)
                    expansion = underlying_graph.get_expansion()
                    if type(expansion) is not str:
                        if expansion > max_expansion: max_expansion = expansion

        return max_expansion

    def __get_gamma(self):
        RW_HDX_status = self.__get_random_walk_similarity()
        worst_exp = 10e-999
        for key in RW_HDX_status.keys():
            if RW_HDX_status[key] > worst_exp:
                worst_exp = RW_HDX_status[key]
        return worst_exp

    def __get_epsilon(self):
        fm = self.__get_flattened_dict()
        score = 0
        total = 0
        for key in self.__get_keys():
            graph = fm[key]
            if graph.get_expansion(walk=True, criterion=1e-4, boolean=True):
                score += 1
            total += 1
        return score / total

    def __get_complex_random_walk_HDX(self):
        # Executing the random walk on the simplicial complex using the up-down operators.
        # Note: No calculations are made in this method, it simply retrieves results from dedicated methods.

        RW_HDX_status = self.__get_random_walk_similarity()
        markov_operator = self.__get_markov_operators()
        lower_operator = self.__get_lower_rw_operators()
        ret_str = "Random Walk Simulation on X: \n"
        worst_exp = 10e-999

        for key in RW_HDX_status.keys():
            if RW_HDX_status[key] > worst_exp:
                worst_exp = RW_HDX_status[key]
            ret_str += "\nSimulating Random Walks on X(" + str(key) + "): \n" + "Markov Operator (M+) on X(" + \
                       str(key) + "): \n" + str(markov_operator[key]) + "\nLower Random Walk Operator (UD) on X(" + \
                       str(key) + "): \n" + str(lower_operator[key]) + "\n||M+ - UD||op = " + \
                       str(RW_HDX_status[key]) + ".\n"

        ret_str += "\nThe Simplicial Complex, X, is a " + str(worst_exp) + "-HDX by the gamma-HDX definition."
        return ret_str

    def __get_simplex_dict(self, simplices=None):
        # Constructing the dictionary storing the simplicial complex in format:
        # { X(1) : [(edge1), (edge2), ...]
        #   X(2) : [(triangle1), (triangle2), ...] }
        __simplex_dict = {}

        if simplices is None:
            simplices = self.simplices()

        # Adding elements from simplices object to the respective key dimension of the dictionary
        for simplex in simplices:
            simplex_dim = len(simplex)
            __simplex_dict.setdefault(simplex_dim, list())
            __simplex_dict[simplex_dim].append(simplex)

        __simplex_dict = self.__get_implied_simplices(__simplex_dict)

        return __simplex_dict

    def __get_implied_simplices(self, __simplex_dict):
        # Adding all implied edges from higher dimensional edges - i.e. a triangle (i,j,k) will imply the edges
        # (i,j), (i,k) and (j,k) and vertices (i,), (j,) and (k,)
        # mode == 1 is used for getting implied simplices for the whole hyper graph, and mode == 2 is used
        # to get implied simplices of the underlying graphs.
        new_simplex = set()
        dim = self.__get_simplex_dim(__simplex_dict)

        for i in range(dim, 1, -1):
            new_simplex.clear()
            for simplex in __simplex_dict[i]:
                t = len(simplex) - 1
                while t > 0:
                    # Finding all possible edge-combinations of a simplex in lower dimensions
                    comb = iter.combinations(simplex, t)
                    for item in comb:
                        new_simplex.add(item)
                    t -= 1
            for new_item in new_simplex:
                k = len(new_item)
                __simplex_dict.setdefault(k, list())
                included = set(__simplex_dict[k])
                if new_item not in included:
                    __simplex_dict[k].append(new_item)

        return __simplex_dict

    def __get_simplex_dim(self, __simplex_dict=None):
        # Calculating the dimension of __simplex_dict (which is the whole complex by default)
        max_dim = 1
        if __simplex_dict is None:
            for simplex in self.simplices():
                if len(simplex) > max_dim:
                    max_dim = len(simplex)
        else:
            for key in __simplex_dict.keys():
                if key > max_dim:
                    max_dim = key
        return max_dim

    def __init_empty_tensor(self, size):
        # Initialise an empty tensor of size=size.
        k = []
        dim = self.__get_simplex_dim()
        for i in range(dim):
            k.append(size)
        k = tuple(k)
        return np.zeros(k)

    def __convert_simplices(self, __simplex_dict):
        # Converting all lower dimensional simplices to their respective dim tuples, i.e. lines, dots etc. to
        # triangles - e.g. an edge (i,j) will be stored in X at index (i,j,j) and (i,j,i)
        # Used for finding the indices to the adjacency tensor.
        dim = self.__get_simplex_dim()
        new_simplex = set()

        for i in range(dim - 1, 0, -1):
            new_simplex.clear()
            for simplex in __simplex_dict[i]:
                for item in simplex:
                    t = len(simplex)
                    temp = list(simplex)
                    while t < dim:
                        temp.append(item)
                        t += 1
                    temp = tuple(temp)
                    new_simplex.add(temp)
            # Removing the tuples of lower dimensional representations and replacing them with appropriate higher
            # dimensional representations. i.e. remove (i,j) and add (i,j,j), (i,j,i)
            included = set(__simplex_dict[i])
            del __simplex_dict[i]
            __simplex_dict.setdefault(i, list())
            for new_item in new_simplex:
                if new_item not in included:
                    __simplex_dict[i].append(new_item)

        # Finding permutations of all simplices and adding them to the dictionary - i.e. for index (i,j,k), we generate
        # (i,k,j), (j,i,k), (j,k,i), (k,j,i) and (k,i,j)
        for i in range(dim, 0, -1):
            new_simplex.clear()
            for simplex in __simplex_dict[i]:
                perm = iter.permutations(simplex)
                for pos in perm:
                    new_simplex.add(pos)
            included = set(__simplex_dict[i])
            for new_item in new_simplex:
                if new_item not in included:
                    __simplex_dict[i].append(new_item)

        return __simplex_dict

    def __get_tensor(self):
        # Builds the tensor X where x(m) = 1 iff m is a simplex in the complex. Note: m is of the dimension
        # same as the complex itself. E.g. for a 3d complex, x(i,j,k) = 1 if (i,j,k) is a triangle.
        size = self.vertex_count()
        dim = self.__get_simplex_dim()
        X = self.__init_empty_tensor(size)
        __simplex_dict = self.__get_simplex_dict()
        __simplex_dict = self.__convert_simplices(__simplex_dict)

        for i in range(dim, 0, -1):
            for simplex in __simplex_dict[i]:
                X[simplex] = 1
        return X

    def __get_underlying_graph(self, simplices):
        # Returns the underlying graph (edges and vertices) of simplices.
        __simplex_dict = self.__get_simplex_dict(simplices)
        __simplex_dim = self.__get_simplex_dim(__simplex_dict)
        for i in range(2, __simplex_dim):
            del __simplex_dict[i + 1]

        __simplex_dict.setdefault(2, list())
        return __simplex_dict

    def __get_link(self, vertex):
        # Compute and return the simplices that are included in the link of vertex.
        if type(vertex) is not tuple:
            raise TypeError('Vertex must be input as a tuple.')

        dim = self.__get_simplex_dim()
        simplex_dim = len(vertex)

        if dim < (simplex_dim + 2):
            ret_str = "Simplices with dimension greater than (SC dimension - 2) do not have any links. In this case, " \
                      "the simplex dimension is " + str(simplex_dim) + " and SC dimension is " + str(dim) + "."
            return ret_str

        links = []
        __simplex_dict = self.__get_simplex_dict()
        vertex_set = set(vertex)
        for key in range(simplex_dim + 1, dim + 1):
            for simplex in __simplex_dict[key]:
                temp_set = set(simplex)
                if vertex_set.issubset(temp_set):
                    links.append(tuple(temp_set - vertex_set))
        return links

    def __get_local_HDX_status(self):
        # Create a summary of the local expansion status for the simplicial complex by computing lambda2 (when possible)
        # and the random walk expansion for each underlying graph.

        ret_str = ""
        base_dict = self.__get_underlying_graph(self.simplices())
        base_graph = Graph(base_dict)
        base_adjacency = base_graph.normalised_adjacency()
        ret_str += "Underlying graph of X is, G(1) = " + str(base_dict) + "\n" + str(base_adjacency) + "\n"
        max_expansion = 1e-99
        expansion = base_graph.get_expansion()
        if type(expansion) is not str:
            if expansion > max_expansion: max_expansion = expansion
            rw_expansion = base_graph.get_expansion(walk=True)
            ret_str += "lambda(G1) = " + str(expansion) + "\n"
        else:
            rw_expansion = expansion
        ret_str += str(rw_expansion) + "\n\n"

        i = 2
        __simplex_dict = self.__get_simplex_dict()
        max_dim = self.__get_simplex_dim()
        # ret_str += "Maximum dimension is " + str(max_dim) + "hence the link calculation loop should end with key = " \
        #          + str(max_dim - 2) + "\n\n"

        for key in __simplex_dict.keys():
            if key > max_dim - 2:
                continue
            ret_str += "Computing local expansion for X(" + str(key - 1) + ")\n\n"
            for simplex in __simplex_dict[key]:
                link = self.__get_link(simplex)
                if len(link) > 0:
                    underlying_graph_dict = self.__get_underlying_graph(link)
                    underlying_graph = Graph(underlying_graph_dict)
                    adjacency_matrix = underlying_graph.normalised_adjacency()
                    expansion = underlying_graph.get_expansion()

                    if type(expansion) is not str:
                        if expansion > max_expansion: max_expansion = expansion
                        rw_expansion = base_graph.get_expansion(walk=True)
                    else:
                        rw_expansion = expansion

                    ret_str += "The link for the simplex, S = {" + str(simplex) + "} is: \nXs = " + str(link) + \
                               "\nThe link has an underlying graph, G(" + str(i) + ")= " + \
                               str(underlying_graph_dict) + "\nThe underlying graph has an adjacency matrix: \n" + \
                               str(adjacency_matrix) + "\n" + str(rw_expansion)

                    if type(expansion) is not str:
                        ret_str += "\n lambda(G" + str(i) + ") = " + str(expansion) + "\n\n"
                    else:
                        ret_str += "\n\n"
                    i += 1
                else:
                    ret_str += "The simplex, S = {" + str(simplex) + \
                               "} does not have a link and has no underlying graph." + "\n\n"

        ret_str += "The Simplicial Complex, X, is a " + str(max_expansion) + "-HDX by the lambda-HDX definition."

        self.plot_underlying_link_graphs()
        return ret_str

    def __get_keys(self):
        # Returns the keys to the dictionaries storing flatenned adjacency matrices
        max_dim = self.__get_simplex_dim()
        return list(range(2, max_dim + 1))

    def __init_vertex_dict(self):
        # Initialise an empty vertex dictionary
        __vertex_dict = {}
        for key in self.__get_keys():
            __vertex_dict.setdefault(key, {})
        return __vertex_dict

    def __init_adjacency_matrix_dict(self):
        # Initialise an empty dictionary containing empty adjacency matrices of the appropriate size.
        __simplex_dict = self.__get_simplex_dict()
        __matrix_dict = {}
        keys = self.__get_keys()
        for key in keys:
            size = len(__simplex_dict[key - 1])
            __matrix_dict.setdefault(key, np.zeros([size, size]))
        return __matrix_dict

    def __get_vertex_dict(self):
        # Creating __vertex_dict: a dictionary of dictionaries which allocates an integer index to each vertex
        # (including 'vertices' from higher dimensions, for e.g. the (1,2) in dimension 3 matrix would be stored as
        # the vertex 0 etc.).
        __vertex_dict = self.__init_vertex_dict()
        __simplex_dict = self.__get_simplex_dict()
        for key in self.__get_keys():
            index = 0
            for simplex in __simplex_dict[key - 1]:
                __vertex_dict[key].setdefault(simplex, index)
                index += 1
        return __vertex_dict

    def __get_adjacency_matrix_dict(self):
        # Generating combos: a list of possible combinations that can be generated from each simplex.
        # Completing __matrix_dict: a dictionary of np.ndarray adjacency matrices for the flattened graphs
        combos = []
        __vertex_dict = self.__get_vertex_dict()
        __matrix_dict = self.__init_adjacency_matrix_dict()
        __simplex_dict = self.__get_simplex_dict()

        for key in self.__get_keys():
            np.fill_diagonal(__matrix_dict[key], 1)

            # Finding the set of possible combinations of each hyperedge in a dimension lower, i.e. for the edge (1,
            # 2,3,4) combos would be {(1,2,3), (1,2,4), (1,3,4), (2,3,4)}, and appending each of these sets into
            # combos.
            for simplex in __simplex_dict[key]:
                combos.append(set(iter.combinations(simplex, key - 1)))

            # For each hyperedge, finding all possible pairs of edges in a dimension lower, i.e. for the edge (1,2,3),
            # simplex_pairs would be {(1,2)&(2,3), (1,2)&(1,3), (2,3)&(1,3)}.
            simplex_pairs = iter.combinations(__simplex_dict[key - 1], 2)

            # For each pair, we check whether it is a subset of a set of combinations of a hyperedge from a higher
            # dimension.
            for pair in simplex_pairs:
                # temporary set variable pair_set is created to use the issubset() method to check whether the pair
                # exists within combos. Original pair is retained for indexing purposes.
                pair_set = set(pair)
                for item in combos:
                    if pair_set.issubset(item):
                        # Getting the vertex allocation the found pair that shares an edge
                        # and reflect the found edge on matrix_dict
                        e1 = __vertex_dict[key].get(pair[0])
                        e2 = __vertex_dict[key].get(pair[1])
                        # print("vertex: ", e1, ":", pair[0] ," and vertex: ", e2, ":", pair[1] , " share an edge.")
                        __matrix_dict[key][e1][e2] = 1
                        __matrix_dict[key][e2][e1] = 1
        return __matrix_dict

    def __get_flattened_dict(self):
        # Converts __matrix_dict to a dictionary of dictionaries containing information compatible with the
        # Graph object. Allows us to implement a flattened matrix as Graph and attain its graph.
        __matrix_dict = self.__get_adjacency_matrix_dict()
        temp_dict = {}
        ret_dict = {}
        for key in self.__get_keys():
            temp_dict[key] = {1: set(), 2: set()}
            adjacency = __matrix_dict[key]
            for i in range(len(adjacency)):
                for j in range(len(adjacency)):
                    if adjacency[i][j] == 1:
                        temp_dict[key][1].add((i,))
                        temp_dict[key][1].add((j,))
                        temp_dict[key][2].add((i, j))

            graph_dict = temp_dict[key]
            ret_dict[key] = Graph(graph_dict)
        return ret_dict

    def __get_flattened_HDX(self):
        # Compute and summarise the HDX status by executing random walks on the flattened graphs
        fm = self.__get_flattened_dict()
        self.flattened_matrices()
        ret_str = "Calculating Expansion of Flattened Matrices:"
        for key in self.__get_keys():
            graph = fm[key]
            expansion = graph.get_expansion(walk=True, criterion=1e-4)
            ret_str += "\n\nExpansion for X(" + str(key - 1) + ")\n" + str(expansion)
        return ret_str

    def __get_graph_count(self):
        # Returns the number of underlying graphs for links that need plotting.
        # The count does not include links that don't have an underlying graph, or have a single vertex graph.
        count = 1
        __simplex_dict = self.__get_simplex_dict()
        max_dim = self.__get_simplex_dim()
        for key in __simplex_dict.keys():
            if key > max_dim - 2:
                continue
            for simplex in __simplex_dict[key]:
                link = self.__get_link(simplex)
                if len(link) > 0:
                    count += 1
        return count

    def __plot_underlying_graphs(self):
        # Getting the underlying graphs of all the links using the Graph object and networkX plotting
        # Graphs are arranged in a grid of 3 columns
        fig = plt.figure(figsize=(11.69, 8.27))
        rows = (self.__get_graph_count() // 3 + 1)
        i = 1
        title = "G(" + str(i) + "), for the link of {}"
        labels = []

        base_dict = self.__get_underlying_graph(self.simplices())
        base_graph = Graph(base_dict)
        for vertex in base_dict[1]:
            labels.append(vertex[0])

        ax = fig.add_subplot(rows, 3, i, title=title)
        base_graph.display_graph(ax=ax)

        __simplex_dict = self.__get_simplex_dict()
        max_dim = self.__get_simplex_dim()
        for key in __simplex_dict.keys():
            if key > max_dim - 2:
                continue
            for simplex in __simplex_dict[key]:
                link = self.__get_link(simplex)
                if len(link) > 0:
                    i += 1
                    title = "G(" + str(i) + "), for the link of {" + str(simplex) + "}"
                    labels = []
                    underlying_graph_dict = self.__get_underlying_graph(link)
                    for vertex in underlying_graph_dict[1]:
                        labels.append(vertex[0])
                    underlying_graph = Graph(underlying_graph_dict)
                    ax = fig.add_subplot(rows, 3, i, title=title)
                    underlying_graph.display_graph(ax=ax, labels=labels)

        plt.tight_layout()
        savename = "UnderlyingGraphs" + str(random.randint(1, 1001))
        plt.savefig('plots/%s.png' % str(savename))
        plt.show()

    def __get_flattened_graphs(self):
        # Plot the flattened graphs with appropriate node labels on a grid with 2 columns
        fm = self.__get_flattened_dict()
        __vertex_dict = self.__get_vertex_dict()
        fig = plt.figure(figsize=(11.69, 8.27))
        rows = (len(fm) // 2 + 1)

        for key in self.__get_keys():
            title = "Graph for dimension " + str(key - 1)
            labels = []
            graph = fm[key]
            for vertex in __vertex_dict[key].keys():
                if len(vertex) > 1:
                    labels.append(vertex)
                else:
                    labels.append(vertex[0])

            ax = fig.add_subplot(rows, 2, key - 1, title=title)
            graph.display_graph(ax=ax, labels=labels)

        plt.tight_layout()
        savename = "FlattenedComplex" + str(random.randint(1, 1001))
        plt.savefig('plots/%s.png' % str(savename))
        plt.show()

    def __get_weighted_distribution(self):
        # Compute the weighted distribution as described Definition 3.3.2 on the final report.
        max_dim = self.__get_simplex_dim()
        __simplex_dict = self.__get_simplex_dict()
        distribution = {}
        counts = {}
        base_probability = 1 / len(__simplex_dict[max_dim])
        for face in __simplex_dict[max_dim]:
            distribution.setdefault(max_dim, {})
            counts.setdefault(max_dim, {})
            distribution[max_dim][face] = base_probability
            counts[max_dim][face] = len(__simplex_dict[max_dim])

        for i in range(max_dim - 1, 0, -1):
            distribution.setdefault(i, {})
            counts.setdefault(i, {})
            for face in __simplex_dict[i]:
                count = 0
                face = set(face)
                for element in __simplex_dict[i + 1]:
                    element = set(element)
                    if face.issubset(element):
                        count += 1
                denominator = (i + 1) * len(distribution[i + 1])
                distribution[i][tuple(face)] = count / denominator
                counts[i][tuple(face)] = count

        return distribution, counts

    def __get_down_operators(self):
        # Compute all the down operators and store them all in a dictionary with keys representing the dimension.
        # i.e. D0, D1, ..., Dd
        max_dim = self.__get_simplex_dim()
        distribution, counts = self.__get_weighted_distribution()
        pi0 = list(distribution[1].values())
        down_operators = {0: np.array([pi0, ] * len(pi0))}

        for k in range(1, max_dim):
            rows = list(counts[k].keys())
            columns = list(counts[k + 1].keys())
            temp_matrix = np.zeros([len(rows), len(columns)])

            for i in range(len(rows)):
                for j in range(len(columns)):
                    a = set(rows[i])
                    b = set(columns[j])
                    if a.issubset(b):
                        temp_matrix[i][j] = 1 / counts[k][rows[i]]

            down_operators.setdefault(k, temp_matrix)

        return down_operators

    def __get_up_operators(self):
        # Compute all the up operators and store them all in a dictionary with keys representing the dimension.
        # i.e. U0, U1, ..., Ud-1
        max_dim = self.__get_simplex_dim()
        _, counts = self.__get_weighted_distribution()
        up_operators = {}

        for k in range(max_dim - 1):
            columns = list(counts[k + 1].keys())
            rows = list(counts[k + 2].keys())
            temp_matrix = np.zeros([len(rows), len(columns)])

            for i in range(len(rows)):
                for j in range(len(columns)):
                    a = set(rows[i])
                    b = set(columns[j])
                    if b.issubset(a):
                        temp_matrix[i][j] = 1 / (k + 2)

            up_operators.setdefault(k, temp_matrix)

        return up_operators

    def __get_markov_operators(self):
        # Compute all the Markov operators and store them all in a dictionary with keys representing the dimension.
        # i.e. M+0, M+1, ..., M+d-1
        # M+i = Di+1Ui - Id
        up = self.__get_up_operators()
        down = self.__get_down_operators()
        markov_operators = {}

        for i in up.keys():
            markov_operators[i] = np.matmul(down[i + 1], up[i])
            np.fill_diagonal(markov_operators[i], 0)

        return markov_operators

    def __get_lower_rw_operators(self):
        # Compute all the lower random walk operators and store them all in a dictionary with keys representing
        # the dimension. i.e. Jpi = U-1D0, U0D1, ..., Ud-2Dd-1
        up = self.__get_up_operators()
        down = self.__get_down_operators()
        lower_rw = {0: down[0]}

        for i in up.keys():
            if i == 0: continue
            lower_rw[i] = np.matmul(up[i - 1], down[i])

        return lower_rw

    def __get_random_walk_similarity(self):
        # Compute the gamma-similarity between the non-lazy upper random walk (markov operator) and the lower random
        # walk by finding the operator norm of the difference between the two operators.
        markov_operator = self.__get_markov_operators()
        lower_operator = self.__get_lower_rw_operators()
        operator_norm = {}

        for i in markov_operator.keys():
            A = np.subtract(markov_operator[i], lower_operator[i])
            B = np.matmul(A.T, A)
            operator_norm.setdefault(i, math.sqrt(abs(max(np.linalg.eigvals(B)))))

        return operator_norm

    def __get_complex_tree_guide(self):
        tree_guide = {}
        max_dim = self.__get_simplex_dim() + 1
        simplex_dict = self.__get_simplex_dict()
        vertex = 0

        for key in range(1, max_dim):
            for simplex in simplex_dict[key]:
                tree_guide.setdefault(simplex, (vertex,))
                vertex += 1

        return tree_guide

    def __get_complex_tree(self):
        tree_guide = self.__get_complex_tree_guide()
        tree_dict = {1: list(tree_guide.values()), 2: list()}
        simplex_dict = self.__get_simplex_dict()
        max_dim = self.__get_simplex_dim() + 1

        for key in range(1, max_dim):
            for simplex in simplex_dict[key]:
                if key == max_dim - 1: continue
                for face in simplex_dict[key + 1]:
                    a = set(simplex)
                    b = set(face)
                    if a.issubset(b):
                        tree_dict[2].append((tree_guide[simplex][0], tree_guide[face][0]))
        return Graph(tree_dict)

    def __plot_complex_tree(self):
        tree = self.__get_complex_tree()
        guide = self.__get_complex_tree_guide()

        title = "ComplexTree" + str(random.randint(1, 1001))
        tree.display_graph(labels=list(guide.keys()), title=title)
        return
