import chinesepostman as cp
import planaridade as pl
import greedy_coloring as color
import networkx as nx
import numpy as np


__author__ = "Résmony Muniz"


def get_graph(file):
    m = []
    with open(file, 'r') as g:
        for line in g:
            line = line.rstrip()
            data = line.split('\t')
            str_to_int = [int(n) for n in data]
            # populate matrix
            m.append(str_to_int)
    g.close()
    return m


if __name__ == '__main__':
    m, n = get_graph('graphs_adj_matrix'), get_graph('chinesepostman_adj')
    g, h = np.matrix(m), np.matrix(n)
    G, H = nx.from_numpy_matrix(g), nx.from_numpy_matrix(h)

    print('Arestas G:', G.edges)
    print('Arestas H:', H.edges)

    '''Chinese Postman'''
    print('\nCiclo Chinese Postman', cp.chinese_postman(H))

    '''Checar planaridade'''
    print('G é planar?', pl.check_planaridade(G))

    '''Coloração'''
    print('(Vértice: Cor)', color.greedy_coloring(G, strategy='mais_largo'))
