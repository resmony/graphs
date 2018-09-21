import networkx as nx
from itertools import combinations

__author__ = 'Résmony Muniz'


def grafo_impar(graph):
	resultado = nx.Graph()
	vertices_impar = [n for n in graph.nodes() if graph.degree(n) % 2 == 1]

	for u in vertices_impar:
		caminhos = nx.shortest_path(graph, source=u, weight='weight')
		comprimento = nx.shortest_path_length(graph, source=u, weight='weight')
		for v in vertices_impar:
			if u <= v:
				continue

			resultado.add_edge(u, v, weight=-comprimento[v], path=caminhos[v])

	return resultado


def edge_sum(graph):
	total = 0
	for u, v, data in graph.edges(data=True):
		total += data['weight']
	return total


def pares(lista):
	i = iter(lista)
	pri, pre, item = i.__next__(), i.__next__(), i.__next__()
	for item in i:
		yield pre, item
		pre = item


def custo_combinacoes(graph, combinacao):
	custo = 0
	for u, v in combinacao:
		if v <= u:
			continue
		d = graph[u][v]
		custo += abs(d['weight'])
	return custo


def encontrar_combinacoes(graph):
	melhor_combinacao = nx.max_weight_matching(graph, True)
	combinacoes = [melhor_combinacao]

	for u, v in melhor_combinacao:
		if v <= u:
			continue

		g_menor = nx.Graph(graph)
		g_menor.remove_edge(u, v)
		combinacao = nx.max_weight_matching(g_menor, True)
		if len(combinacao) > 0:
			combinacoes.append(combinacao)

	custo_combs = [(custo_combinacoes(graph, combinacao), combinacao) for combinacao in combinacoes]
	custo_combs.sort()

	# remover combinacoes com memso custo duplicadas
	combinacoes_finais = []
	ultimo_custo = None
	for custo, comb in custo_combs:
		if custo == ultimo_custo:
			continue
		ultimo_custo = custo
		combinacoes_finais.append((custo, comb))

	return combinacoes_finais


def construir_grafo_euleriano(graph, impar, combinacao):
	grafo_euleriano = nx.MultiGraph(graph)

	for u, v in combinacao:
		if v <= u:
			continue
		aresta = impar[u][v]
		caminho = aresta['path']

		for p, q in pares(caminho):
			grafo_euleriano.add_edge(p, q, weight=graph[p][q]['weight'])

	return grafo_euleriano


def eulerize(G):
	"""
	Transforma o grafo em um grafo euleriano
	"""
	if G.order() == 0:
		raise nx.NetworkXPointlessConcept("Graph null")
	if not nx.is_connected(G):
		raise nx.NetworkXError("G é desconexo")
	vertices_grau_impar = [n for n, d in G.degree() if d % 2 == 1]
	G = nx.MultiGraph(G)
	if len(vertices_grau_impar) == 0:
		return G

	# caminho mais curto entre pares de vértices com grau ímpar
	pares_caminho_grau_impar = [(m, {n: nx.shortest_path(G, source=m, target=n)})
								for m, n in combinations(vertices_grau_impar, 2)]

	H = nx.Graph()
	for n, p in pares_caminho_grau_impar:
		for m, q in p.items():
			if n != m:
				H.add_edge(m, n, weight=1 / len(q), path=q)

	melhor_comb = nx.Graph(list(nx.max_weight_matching(H)))

	for m, n in melhor_comb.edges():
		caminho = H[m][n]['path']
		G.add_edges_from(nx.utils.pairwise(caminho))
	return G


def circuito_euleriano(graph):
	G = None
	# print("graph é euleriano? ", nx.is_eulerian(graph))
	if not nx.is_eulerian(graph):
		G = eulerize(graph)

	circuito = list(nx.eulerian_circuit(G))
	nos = []
	for u, v in circuito:
		nos.append(u)

	nos.append(circuito[0][0])
	return nos


def chinese_postman(graph):
	# O(V'*(E + V log(V)) )
	impar = grafo_impar(graph)

	# O(V'^3)
	combinacao = nx.max_weight_matching(impar, True)
	# print('soma ímpar: ', edge_sum(impar))

	# O(E)
	grafo_euleriano = construir_grafo_euleriano(graph, impar, combinacao)
	nos = circuito_euleriano(grafo_euleriano)

	# lista com o caminho
	return nos
