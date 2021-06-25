# Author: Mihir Tare
# Created to fulfill the requirements for CS350: Data Science Third Year Project
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eig
import random


class Graph(object):

    def __init__(self, graph_dict=None):
        # Uses an empty dictionary to init Graph object if none provided.
        if graph_dict is None:
            graph_dict = {1: list(), 2: list()}
        self.__graph_dict = graph_dict
        self.__size = len(graph_dict[1])

    def vertices(self):
        return self.__graph_dict[1]

    def vertex_count(self):
        return self.__size

    def edges(self):
        return self.__get_edges()

    def edge_count(self):
        return len(self.edges())

    def adjacency(self):
        # Getter method for the adjacency matrix
        return self.__get_adjacency()

    def normalised_adjacency(self):
        # Getter method for the normalised adjacency matrix
        return self.__get_normalised_adjacency()

    def laplacian(self):
        # Getter method for the laplacian matrix
        laplacian, _, _ = self.__get_laplacian()
        return laplacian

    def get_expansion(self, walk=False, criterion=1e-8, boolean=False):
        # Retrieve the expansion properties of the graph based on configuration.
        # i.e. either random walk expansion, lambda expansion (if d-regular)
        regularity = self.__get_regularity()
        if walk is True or regularity is False:
            if boolean is True:
                return self.__get_walk_expansion(criterion)[1]
            else:
                return self.__get_walk_expansion(criterion)[0]
        else:
            return self.__get_lambda()

    def cheeger(self):
        # Compute the cheeger constant for a graph
        laplacian, _, _ = self.__get_laplacian()
        ub, lb = self.__get_cheeger_bounds()
        ret_str = "Cheeger constant is " + str(self.__get_cheeger()) + " with approximated upper bound " + \
                  str(ub) + ", and lower bound " + str(lb) + \
                  " and laplacian \n" + str(laplacian)
        return ret_str

    def display_graph(self, ax=None, labels=None, title=None):
        # Method to display a graph (compatible with SComplex)
        if labels is None:
            labels = range(self.__size)

        if title is None:
            title = "Graph" + str(random.randint(1,1001))

        if ax is None:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(1, 1, 1, title=title)
            self.__plot_graph(labels, ax)
            plt.tight_layout()
            plt.savefig('plots/%s.png' % str(title))
            plt.show()
            return
        else:
            return self.__plot_graph(labels, ax)

    def __get_edges(self):
        return self.__graph_dict[2]

    def __standardise_adjacency(self):
        # Converts the edges and vertices to a format that is incremental in indices to ease matrix construction;
        # e.g. if G = {(1), (4), (5), (4,5)}, the standardised adjacency would be G = {(0), (1), (2), (1,2)}
        __adjacency_dict = {}
        i = 0
        seen = set()
        ret = {1: list(), 2: list()}

        for vertex in self.__graph_dict[1]:
            __adjacency_dict[vertex[0]] = i
            if (i,) not in seen:
                seen.add((i,))
                ret[1].append((i,))
                i += 1
        for edge in self.__graph_dict[2]:
            e1 = edge[0]
            e2 = edge[1]
            v1 = __adjacency_dict[e1]
            v2 = __adjacency_dict[e2]
            if (v1, v2) not in seen:
                # In the case where an edge connects vertices that were not included in the input dictionary
                if (v1,) not in seen:
                    seen.add((v1,))
                    ret[1].append((v1,))
                if (v2,) not in seen:
                    seen.add((v2,))
                    ret[1].append((v2,))
                seen.add((v1, v2))
                ret[2].append((v1, v2))
        return ret

    def __get_adjacency(self):
        # Constructing the adjacency matrix based on the definition
        matrix = np.zeros([self.__size, self.__size])
        np.fill_diagonal(matrix, 1)
        graph_dict = self.__standardise_adjacency()
        for edge in graph_dict[2]:
            v1 = edge[0]
            v2 = edge[1]
            matrix[v1][v2] = 1
            matrix[v2][v1] = 1
        return matrix

    def __get_normalised_adjacency(self):
        # Constructing the normalised adjacency based on the definition
        matrix = self.__get_adjacency()
        normalised = np.zeros([self.__size, self.__size])
        for i in range(len(matrix)):
            d = np.count_nonzero(matrix[i])
            for j in range(len(matrix)):
                if matrix[i][j] == 1:
                    normalised[i][j] = 1 / d
        return normalised

    def __get_laplacian(self):
        # Constructing the laplacian matrix
        adjacency = self.__get_adjacency()
        laplacian = -adjacency
        degree = []
        for i in range(len(adjacency)):
            degree.append(np.count_nonzero(adjacency[i]))

        np.fill_diagonal(laplacian, degree)
        return laplacian, max(degree), min(degree)

    def __get_lambda(self):
        # Returns second largest eigenvalue of normalised adjacency matrix
        adjacency = self.__get_normalised_adjacency()
        if adjacency.shape[0] == 1:
            return 1
        e_values = np.linalg.eigvals(adjacency)
        e_values = -np.sort(-e_values)
        return e_values[1]

    def __get_cheeger_bounds(self):
        # Compute upper and lower bounds on the Cheeger constant using the laplacian matrix
        laplacian, d, _ = self.__get_laplacian()
        e_values = np.linalg.eigvals(laplacian)
        e_values = np.sort(e_values)
        lambda2 = e_values[1]
        lb = lambda2 / 2
        ub = math.sqrt(2 * d * lambda2)
        return ub, lb

    def __get_cheeger(self):
        # Compute the Cheeger constant
        S = self.__get_subsets()
        min_hg = 10e99
        for subset in S:
            e_count = self.__get_outgoing_edges(subset)
            s_size = len(subset)
            hg = (e_count / s_size)
            if hg < min_hg:
                min_hg = hg
        return min_hg

    def __get_subsets(self):
        # Calculates all possible set of vertices of size <= floor(|V|/2)
        matrix = self.__get_adjacency()
        n = np.floor(len(matrix) / 2)
        S = [[]]
        for vertex in range(len(matrix)):
            temp_sets = []
            for subset in S:
                if len(subset) < n:
                    temp_sets.extend([subset + [vertex]])
            S.extend(temp_sets)
        S.pop(0)
        return S

    def __get_outgoing_edges(self, element):
        # Calculates the number of outgoing edges from set of vertices S to its complement.
        matrix = self.__get_adjacency()
        edge_count = 0
        for i in element:
            for j in range(len(matrix[i])):
                if (j not in element) and (matrix[i][j]) > 0:
                    edge_count = edge_count + 1
        return edge_count

    def __get_regularity(self):
        # Returns a boolean stating whether the graph is d-regular
        matrix = self.__get_adjacency()
        degree = np.count_nonzero(matrix[0])
        for i in range(1, len(matrix)):
            if np.count_nonzero(matrix[i]) != degree:
                return False
        return True

    def __get_walk_expansion(self, criterion):
        # Simulates |V| random walks on a graph to check if it converges regardless of starting position.
        normalised = self.__get_normalised_adjacency()

        e_values, e_vectors = np.linalg.eig(normalised.T)
        j = np.where(np.abs(e_values - 1.) < 1e-8)[0][0]
        matrix = np.array(e_vectors[:, j].flat)
        stationary = matrix / matrix.sum()
        max_steps = 10 * math.ceil(math.log2(len(normalised)))

        score = 0
        for k in range(len(normalised)):
            converges = False
            # mu is the distribution of the starting probabilities. Ensures each random walk begins
            # on a different vertex.
            mu = np.zeros(len(normalised))
            mu[k] = 1
            distribution = np.dot(mu, normalised)
            for i in range(max_steps):
                distribution = np.matmul(distribution, normalised)
                mse = ((distribution - stationary) ** 2).mean()
                if mse < criterion:
                    converges = True
                    break
            if converges:
                score += 1

        if score < len(normalised):
            return "Random walk did not converge to the stationary distribution, " + str(stationary) + \
                   ", after O(log(|V|)) iterations on every iteration. Convergence score: " + str(score) + \
                   " times out of " + str(len(normalised)), False
        else:
            return "Random walk converged to stationary distribution, " + str(stationary) + ", in every random walk.", \
                   True

    def __plot_graph(self, labels, ax):
        # Method to plot individual graphs using NetworkX on a specified Axes. ax.
        adjacency = self.__get_adjacency()
        rows, cols = np.where(adjacency == 1)
        label_dict = {}
        for i, label in enumerate(labels):
            label_dict[i] = label
            # Decides the size of the vertex on the plot (to make sure the vertex label doesn't overflow)
            if type(label) is tuple:
                size = 1000 * (len(label))
            else:
                size = 500

        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        pos = nx.circular_layout(gr)

        nx.draw(gr, pos=pos, ax=ax, node_size=size, labels=label_dict, with_labels=True)