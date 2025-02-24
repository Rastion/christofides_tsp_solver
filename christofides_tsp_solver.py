from qubots.base_optimizer import BaseOptimizer
import networkx as nx

class ChristofidesTSPSolver(BaseOptimizer):
    """
    Christofides TSP Solver.

    This solver implements the Christofides algorithm to compute an approximate solution
    for the Traveling Salesman Problem (TSP) for metric instances. It follows these steps:
      1. Compute a Minimum Spanning Tree (MST) of the complete graph.
      2. Identify all vertices with odd degree in the MST.
      3. Compute a minimum weight perfect matching on the odd-degree vertices.
      4. Combine the MST and matching to form an Eulerian multigraph.
      5. Find an Eulerian circuit and shortcut it to form a Hamiltonian circuit (TSP tour).
    """
    def __init__(self):
        # No tunable parameters in this simple implementation.
        pass

    def optimize(self, problem, initial_solution=None, **kwargs):
        n = problem.nb_cities
        dist_matrix = problem.dist_matrix

        # Create a complete graph with nodes 0...n-1 and edge weights from the distance matrix.
        G = nx.complete_graph(n)
        for i, j in G.edges():
            G[i][j]['weight'] = dist_matrix[i][j]

        # Step 1: Compute the Minimum Spanning Tree (MST) of the graph.
        T = nx.minimum_spanning_tree(G, weight='weight')

        # Step 2: Identify all vertices with odd degree in the MST.
        odd_degree_nodes = [node for node in T.nodes() if T.degree(node) % 2 == 1]

        # Step 3: Compute a minimum weight perfect matching for the subgraph induced by odd-degree nodes.
        subgraph = G.subgraph(odd_degree_nodes)
        matching = nx.algorithms.matching.min_weight_matching(subgraph, maxcardinality=True, weight='weight')

        # Step 4: Combine MST and matching edges into a multigraph.
        multigraph = nx.MultiGraph()
        multigraph.add_nodes_from(T.nodes())
        multigraph.add_edges_from(T.edges(data=True))
        for u, v in matching:
            multigraph.add_edge(u, v, weight=G[u][v]['weight'])

        # Step 5: Find an Eulerian circuit in the multigraph (starting from node 0).
        circuit_edges = list(nx.eulerian_circuit(multigraph, source=0))

        # Step 6: Shortcut the Eulerian circuit to form a TSP tour.
        tour = [0]
        for _, v in circuit_edges:
            if v not in tour:
                tour.append(v)
        # Ensure the tour is a closed cycle.
        if tour[0] != tour[-1]:
            tour.append(tour[0])

        cost = self.compute_cost(tour, dist_matrix)
        return tour, cost

    def compute_cost(self, tour, dist_matrix):
        cost = 0
        for i in range(len(tour) - 1):
            cost += dist_matrix[tour[i]][tour[i+1]]
        return cost
