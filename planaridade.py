import networkx as nx
from collections import defaultdict

__author__ = "Résmony Muniz"


def check_planaridade(G):
    planaridade_state = LRPlanaridade(G)
    embedding = planaridade_state.lr_planaridade()
    if embedding is None:
        # graph não é planar
        return False
    else:
        # graph planar
        return True


class Intervalo(object):
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    def empty(self):
        return self.low is None and self.high is None

    def copy(self):
        return Intervalo(self.low, self.high)

    def conflicting(self, b, planaridade_state):
        return (not self.empty() and
                planaridade_state.lowpt[self.high] > planaridade_state.lowpt[b])


class ParConflitante(object):
    def __init__(self, left=Intervalo(), right=Intervalo()):
        self.left = left
        self.right = right

    def trocar(self):
        temp = self.left
        self.left = self.right
        self.right = temp

    def lowest(self, planaridade_state):
        if self.left.empty():
            return planaridade_state.lowpt[self.right.low]
        if self.right.empty():
            return planaridade_state.lowpt[self.left.low]
        return min(planaridade_state.lowpt[self.left.low],
                   planaridade_state.lowpt[self.right.low])


def topo_da_pilha(l):
    if not l:
        return None
    return l[-1]


class LRPlanaridade(object):
    def __init__(self, G):
        self.G = nx.Graph()
        self.G.add_nodes_from(G.nodes)
        for e in G.edges:
            if e[0] != e[1]:
                self.G.add_edge(e[0], e[1])

        self.roots = []

        self.height = defaultdict(lambda: None)

        self.lowpt, self.lowpt2 = {}, {}
        self.nesting_depth = {}

        self.parent_edge = defaultdict(lambda: None)

        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(G.nodes)

        self.adjs, self.ordered_adjs = {}, {}

        self.ref = defaultdict(lambda: None)
        self.side = defaultdict(lambda: 1)

        self.S = []
        self.stack_bottom = {}
        self.lowpt_edge = {}

        self.left_ref, self.right_ref = {}, {}

        self.embedding = PlanarEmbedding()

    def lr_planaridade(self):
        l = self.G.size()
        m = self.G.order()
        if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
            return None

        for v in self.G:
            self.adjs[v] = list(self.G[v])

        for v in self.G:
            if self.height[v] is None:
                self.height[v] = 0
                self.roots.append(v)
                self.dfs_orientation(v)

        self.G, self.lowpt2, self.adjs = None, None, None

        for v in self.DG:
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)])
        for v in self.roots:
            if not self.dfs_testing(v):
                return None

        self.height = None
        self.lowpt = None
        self.S = None
        self.stack_bottom = None
        self.lowpt_edge = None

        for e in self.DG.edges:
            self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]

        self.embedding.add_nodes_from(self.DG.nodes)
        for v in self.DG:
            self.ordered_adjs[v] = sorted(
                self.DG[v], key=lambda x: self.nesting_depth[(v, x)])
            previous_node = None
            
            for w in self.ordered_adjs[v]:
                self.embedding.add_half_edge_cw(v, w, previous_node)
                previous_node = w

        self.DG, self.nesting_depth, self.ref = None, None, None

        for v in self.roots:
            self.dfs_embedding(v)

        self.roots = None
        self.parent_edge = None
        self.ordered_adjs = None
        self.left_ref = None
        self.right_ref = None
        self.side = None

        return self.embedding

    def dfs_orientation(self, v):
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]

            for w in self.adjs[v][ind[v]:]:
                vw = (v, w)

                if not skip_init[vw]:
                    if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
                        ind[v] += 1
                        continue

                    self.DG.add_edge(v, w)

                    self.lowpt[vw] = self.height[v]
                    self.lowpt2[vw] = self.height[v]
                    if self.height[w] is None:
                        self.parent_edge[w] = vw
                        self.height[w] = self.height[v] + 1

                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[vw] = True
                        break
                    else:
                        self.lowpt[vw] = self.height[w]

                self.nesting_depth[vw] = 2 * self.lowpt[vw]
                if self.lowpt2[vw] < self.height[v]:
                    self.nesting_depth[vw] += 1

                if e is not None:
                    if self.lowpt[vw] < self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                        self.lowpt[e] = self.lowpt[vw]
                    elif self.lowpt[vw] > self.lowpt[e]:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
                    else:
                        self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])

                ind[v] += 1

    def dfs_testing(self, v):
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)
        skip_init = defaultdict(lambda: False)

        while dfs_stack:
            v = dfs_stack.pop()
            e = self.parent_edge[v]
            skip_final = False

            for w in self.ordered_adjs[v][ind[v]:]:
                ei = (v, w)

                if not skip_init[ei]:
                    self.stack_bottom[ei] = topo_da_pilha(self.S)

                    if ei == self.parent_edge[w]:
                        dfs_stack.append(v)
                        dfs_stack.append(w)
                        skip_init[ei] = True
                        skip_final = True
                        break
                    else:
                        self.lowpt_edge[ei] = ei
                        self.S.append(ParConflitante(right=Intervalo(ei, ei)))

                if self.lowpt[ei] < self.height[v]:
                    if w == self.ordered_adjs[v][0]:
                        self.lowpt_edge[e] = self.lowpt_edge[ei]
                    else:
                        if not self.add_constraints(ei, e):
                            return False

                ind[v] += 1

            if not skip_final:
               if e is not None:
                    self.remove_back_edges(e)

        return True

    def add_constraints(self, ei, e):
        P = ParConflitante()
        while True:
            Q = self.S.pop()
            if not Q.left.empty():
                Q.trocar()
            if not Q.left.empty():
                return False
            if self.lowpt[Q.right.low] > self.lowpt[e]:
                if P.right.empty():
                    P.right = Q.right.copy()
                else:
                    self.ref[P.right.low] = Q.right.high
                P.right.low = Q.right.low
            else:
                self.ref[Q.right.low] = self.lowpt_edge[e]
            if topo_da_pilha(self.S) == self.stack_bottom[ei]:
                break
        while (topo_da_pilha(self.S).left.conflicting(ei, self) or
               topo_da_pilha(self.S).right.conflicting(ei, self)):
            Q = self.S.pop()
            if Q.right.conflicting(ei, self):
                Q.trocar()
            if Q.right.conflicting(ei, self):
                return False
            self.ref[P.right.low] = Q.right.high
            if Q.right.low is not None:
                P.right.low = Q.right.low

            if P.left.empty():
                P.left = Q.left.copy()
            else:
                self.ref[P.left.low] = Q.left.high
            P.left.low = Q.left.low

        if not (P.left.empty() and P.right.empty()):
            self.S.append(P)
        return True

    def remove_back_edges(self, e):
        u = e[0]
        while self.S and topo_da_pilha(self.S).lowest(self) == self.height[u]:
            P = self.S.pop()
            if P.left.low is not None:
                self.side[P.left.low] = -1

        if self.S:
            P = self.S.pop()
            while P.left.high is not None and P.left.high[1] == u:
                P.left.high = self.ref[P.left.high]
            if P.left.high is None and P.left.low is not None:
                self.ref[P.left.low] = P.right.low
                self.side[P.left.low] = -1
                P.left.low = None
            while P.right.high is not None and P.right.high[1] == u:
                P.right.high = self.ref[P.right.high]
            if P.right.high is None and P.right.low is not None:
                self.ref[P.right.low] = P.left.low
                self.side[P.right.low] = -1
                P.right.low = None
            self.S.append(P)

        if self.lowpt[e] < self.height[u]:
            hl = topo_da_pilha(self.S).left.high
            hr = topo_da_pilha(self.S).right.high

            if hl is not None and (
                            hr is None or self.lowpt[hl] > self.lowpt[hr]):
                self.ref[e] = hl
            else:
                self.ref[e] = hr

    def dfs_embedding(self, v):
        dfs_stack = [v]
        ind = defaultdict(lambda: 0)

        while dfs_stack:
            v = dfs_stack.pop()

            for w in self.ordered_adjs[v][ind[v]:]:
                ind[v] += 1
                ei = (v, w)

                if ei == self.parent_edge[w]:
                    self.embedding.add_half_edge_first(w, v)
                    self.left_ref[v] = w
                    self.right_ref[v] = w

                    dfs_stack.append(v)
                    dfs_stack.append(w)
                    break
                else:
                    if self.side[ei] == 1:
                        self.embedding.add_half_edge_cw(w, v,
                                                        self.right_ref[w])
                    else:
                        self.embedding.add_half_edge_ccw(w, v,
                                                         self.left_ref[w])
                        self.left_ref[w] = v

    def sign(self, e):
        dfs_stack = [e]
        old_ref = defaultdict(lambda: None)

        while dfs_stack:
            e = dfs_stack.pop()

            if self.ref[e] is not None:
                dfs_stack.append(e)
                dfs_stack.append(self.ref[e])
                old_ref[e] = self.ref[e]
                self.ref[e] = None
            else:
                self.side[e] *= self.side[old_ref[e]]

        return self.side[e]


class PlanarEmbedding(nx.DiGraph):
    def get_data(self):
        embedding = dict()
        for v in self:
            embedding[v] = list(self.neighbors_cw_order(v))
        return embedding

    def neighbors_cw_order(self, v):
        if len(self[v]) == 0:
            return
        start_node = self.nodes[v]['first_nbr']
        yield start_node
        current_node = self[v][start_node]['cw']

        while start_node != current_node:
            yield current_node
            current_node = self[v][current_node]['cw']

    def check_structure(self):
        for v in self:
            try:
                sorted_nbrs = set(self.neighbors_cw_order(v))
            except KeyError:
                msg = "Orientação do nó vizinho ausente {}".format(v)
                raise nx.NetworkXException(msg)

            unsorted_nbrs = set(self[v])
            if sorted_nbrs != unsorted_nbrs:
                msg = "Orientação das arestas não definidas corretamente"
                raise nx.NetworkXException(msg)
            for w in self[v]:
                if not self.has_edge(w, v):
                    msg = "Metade a aresta oposta faltando."
                    raise nx.NetworkXException(msg)

        # Check planaridade
        counted_half_edges = set()
        for component in nx.connected_components(self):
            if len(component) == 1:
                continue
            num_nodes = len(component)
            num_half_edges = 0
            num_faces = 0

            for v in component:
                for w in self.neighbors_cw_order(v):
                    num_half_edges += 1
                    if (v, w) not in counted_half_edges:
                        num_faces += 1
                        self.traverse_face(v, w, counted_half_edges)

            num_edges = num_half_edges // 2
            if num_nodes - num_edges + num_faces != 2:
                msg = "Grafo não corresponde a Fórmula de Euler"
                raise nx.NetworkXException(msg)

    def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
        if reference_neighbor is None:
            self.add_edge(start_node, end_node)
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
        else:
            ccw_reference = self[start_node][reference_neighbor]['ccw']
            self.add_half_edge_cw(start_node, end_node, ccw_reference)

            if reference_neighbor == self.nodes[start_node].get('first_nbr', None):
                self.nodes[start_node]['first_nbr'] = end_node

    def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
        self.add_edge(start_node, end_node)  # Add edge to graph

        if reference_neighbor is None:
            # The start node has no neighbors
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
            return

        if reference_neighbor not in self[start_node]:
            raise nx.NetworkXException(
                "Não é possível adicionar aresta. Referência ao vizinho inexistente.")

        cw_reference = self[start_node][reference_neighbor]['cw']
        self[start_node][reference_neighbor]['cw'] = end_node
        self[start_node][end_node]['cw'] = cw_reference
        self[start_node][cw_reference]['ccw'] = end_node
        self[start_node][end_node]['ccw'] = reference_neighbor

    def connect_components(self, v, w):
        self.add_half_edge_first(v, w)
        self.add_half_edge_first(w, v)

    def add_half_edge_first(self, start_node, end_node):
        if start_node in self and 'first_nbr' in self.nodes[start_node]:
            reference = self.nodes[start_node]['first_nbr']
        else:
            reference = None
        self.add_half_edge_ccw(start_node, end_node, reference)

    def next_face_half_edge(self, v, w):
        new_node = self[w][v]['ccw']
        return w, new_node

    def traverse_face(self, v, w, mark_half_edges=None):
        if mark_half_edges is None:
            mark_half_edges = set()

        face_nodes = [v]
        mark_half_edges.add((v, w))
        prev_node = v
        cur_node = w
        incoming_node = self[v][w]['cw']

        while cur_node != v or prev_node != incoming_node:
            face_nodes.append(cur_node)
            prev_node, cur_node = self.next_face_half_edge(prev_node, cur_node)
            if (prev_node, cur_node) in mark_half_edges:
                raise nx.NetworkXException(
                    "Bad planar embedding. Impossible face.")
            mark_half_edges.add((prev_node, cur_node))

        return face_nodes

    def is_directed(self):
        return False