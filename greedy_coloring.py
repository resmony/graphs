import networkx as nx
import itertools

__author__ = "Résmony Muniz"


class Node(object):

    __slots__ = ['node_id', 'color', 'adj_list', 'adj_color']

    def __init__(self, node_id, n):
        self.node_id = node_id
        self.color = -1
        self.adj_list = None
        self.adj_color = [None for _ in range(n)]

    def __repr__(self):
        return "Node_id: {0}, Color: {1}, Adj_list: ({2}), \
            adj_color: ({3})".format(
            self.node_id, self.color, self.adj_list, self.adj_color)

    def atribuir_cor(self, adj_entry, color):
        adj_entry.col_prev = None
        adj_entry.col_next = self.adj_color[color]
        self.adj_color[color] = adj_entry
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry

    def limpar_cor(self, adj_entry, color):
        if adj_entry.col_prev is None:
            self.adj_color[color] = adj_entry.col_next
        else:
            adj_entry.col_prev.col_next = adj_entry.col_next
        if adj_entry.col_next is not None:
            adj_entry.col_next.col_prev = adj_entry.col_prev

    def iter_vizinhos(self):
        adj_node = self.adj_list
        while adj_node is not None:
            yield adj_node
            adj_node = adj_node.next

    def iter_cor_vizinhos(self, color):
        adj_color_node = self.adj_color[color]
        while adj_color_node is not None:
            yield adj_color_node.node_id
            adj_color_node = adj_color_node.col_next


class AdjEntry(object):

    __slots__ = ['node_id', 'next', 'mate', 'col_next', 'col_prev']

    def __init__(self, node_id):
        self.node_id = node_id
        self.next = None
        self.mate = None
        self.col_next = None
        self.col_prev = None

    def __repr__(self):
        return "Node_id: {0}, Next: ({1}), Mate: ({2}), \
            col_next: ({3}), col_prev: ({4})".format(
            self.node_id,
            self.next,
            self.mate.node_id,
            None if self.col_next is None else self.col_next.node_id,
            None if self.col_prev is None else self.col_prev.node_id
        )


def strategia_mais_largo(G, colors):
    return sorted(G, key=G.degree, reverse=True)


def strategia_conjunto_independente(G, colors):
    remaining = set(G)
    while len(remaining) > 0:
        nodes = _conjunto_independente_max(G.subgraph(remaining))
        remaining -= nodes
        for v in nodes:
            yield v


def _conjunto_independente_max(G):
    resultado = set()
    remaining = set(G)
    while remaining:
        G = G.subgraph(remaining)
        v = min(remaining, key=G.degree)
        resultado.add(v)
        remaining -= set(G[v]) | {v}
    return resultado


STRATEGY = {
    'mais_largo': strategia_mais_largo,
    'conjunto_independente': strategia_conjunto_independente
}


def greedy_coloring_with_interchange(G, nodes):
    n = len(G)

    graph = {node_id: Node(node_id, n) for node_id in G}

    for (node1, node2) in G.edges():
        adj_entry1 = AdjEntry(node2)
        adj_entry2 = AdjEntry(node1)
        adj_entry1.mate = adj_entry2
        adj_entry2.mate = adj_entry1
        node1_head = graph[node1].adj_list
        adj_entry1.next = node1_head
        graph[node1].adj_list = adj_entry1
        node2_head = graph[node2].adj_list
        adj_entry2.next = node2_head
        graph[node2].adj_list = adj_entry2

    k = 0
    for node in nodes:
        neighbors = graph[node].iter_neighbors()
        col_used = {graph[adj_node.node_id].color for adj_node in neighbors}
        col_used.discard(-1)
        k1 = next(itertools.dropwhile(
            lambda x: x in col_used, itertools.count()))

        if k1 > k:
            connected = True
            visited = set()
            col1 = -1
            col2 = -1
            while connected and col1 < k:
                col1 += 1
                neighbor_cols = (
                    graph[node].iter_neighbors_color(col1))
                col1_adj = [it for it in neighbor_cols]

                col2 = col1
                while connected and col2 < k:
                    col2 += 1
                    visited = set(col1_adj)
                    frontier = list(col1_adj)
                    i = 0
                    while i < len(frontier):
                        search_node = frontier[i]
                        i += 1
                        col_opp = (
                            col2 if graph[search_node].color == col1 else col1)
                        neighbor_cols = (
                            graph[search_node].iter_neighbors_color(col_opp))

                        for neighbor in neighbor_cols:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                frontier.append(neighbor)

                    connected = len(visited.intersection(
                        graph[node].iter_neighbors_color(col2))) > 0

            if not connected:
                for search_node in visited:
                    graph[search_node].color = (
                        col2 if graph[search_node].color == col1 else col1)
                    col2_adj = graph[search_node].adj_color[col2]
                    graph[search_node].adj_color[col2] = (
                        graph[search_node].adj_color[col1])
                    graph[search_node].adj_color[col1] = col2_adj

                for search_node in visited:
                    col = graph[search_node].color
                    col_opp = col1 if col == col2 else col2
                    for adj_node in graph[search_node].iter_neighbors():
                        if graph[adj_node.node_id].color != col_opp:
                            # Direct reference to entry
                            adj_mate = adj_node.mate
                            graph[adj_node.node_id].clear_color(
                                adj_mate, col_opp)
                            graph[adj_node.node_id].assign_color(adj_mate, col)
                k1 = col1

        graph[node].color = k1
        k = max(k1, k)

        for adj_node in graph[node].iter_neighbors():
            adj_mate = adj_node.mate
            graph[adj_node.node_id].assign_color(adj_mate, k1)

    return {node.node_id: node.color for node in graph.values()}


def greedy_coloring(G, strategy='mais_largo', interchange=False):
    if len(G) == 0:
        return {}
    strategy = STRATEGY.get(strategy, strategy)
    if not callable(strategy):
        raise nx.NetworkXError('Estratégia de coloramento inválida')

    colors = {}
    nodes = strategy(G, colors)
    if interchange:
        return greedy_coloring_with_interchange(G, nodes)
    for u in nodes:
        cores_vizinhas = {colors[v] for v in G[u] if v in colors}
        for color in itertools.count():
            if color not in cores_vizinhas:
                break
        colors[u] = color

    return colors
