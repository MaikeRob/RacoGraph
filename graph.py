"""
Implementação de estrutura de grafo não-direcionado para o sistema de recomendação.
Utiliza lista de adjacência para armazenar conexões entre nós.
"""
from typing import List, Dict, Any


def is_user(node_id: str) -> bool:
    """Verifica se o nó é um usuário (prefixo 'U')."""
    return isinstance(node_id, str) and node_id.startswith("U")


def is_movie(node_id: str) -> bool:
    """Verifica se o nó é um filme (prefixo 'F')."""
    return isinstance(node_id, str) and node_id.startswith("F")


def is_genre(node_id: str) -> bool:
    """Verifica se o nó é um gênero (prefixo 'G')."""
    return isinstance(node_id, str) and node_id.startswith("G")


class Graph:
    """
    Representa um grafo simples.
    Armazena nós (como 'U1', 'F1') e as conexões (arestas) entre eles.
    """

    def __init__(self):
        """
        Inicia o grafo com dicionários vazios.
        'self.nodes' guarda os dados dos nós (ex: tipo, nome).
        'self.adj' guarda as conexões (lista de adjacência).
        """
        self.nodes = {}  # Guarda os atributos dos nós (ex: {'U1': {'type': 'User'}})
        self.adj = {}    # Guarda as conexões (ex: {'U1': [{'node': 'F1', ...}]})

    def get_node(self, node_id):
        """
        Retorna os dados de um nó, se existir.

        :param node_id: O ID do nó (ex: 'U1', 'F1', 'G1').
        :return: Um dicionário com os atributos do nó ou None se não existir.
        """
        return self.nodes.get(node_id)


    def add_node(self, node_id, node_type, **kwargs):
        """
        Adiciona um novo nó ao grafo, se ele não existir.

        :param node_id: O ID único do nó (ex: 'U1', 'F1', 'G1').
        :param node_type: O tipo do nó (ex: 'User', 'Movie', 'Genre').
        :param kwargs: Outros dados (ex: titulo='Nome do Filme').
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = {'type': node_type, **kwargs}
            self.adj[node_id] = [] # Prepara a lista de vizinhos dele

    def add_edge(self, u_id, v_id, weight=1.0, **kwargs):
        """
        Adiciona uma aresta (conexão) entre dois nós.
        O grafo não é direcionado (a conexão vale nos dois sentidos).

        :param u_id: ID do primeiro nó.
        :param v_id: ID do segundo nó.
        :param weight: O "peso" da conexão (ex: uma nota 5.0).
        :param kwargs: Outros dados (ex: relation_type='avaliou').
        """
        # Garante que os nós existem antes de tentar conectá-los
        if u_id not in self.nodes:
            raise ValueError(f"Nó '{u_id}' não existe. Adicione-o primeiro.")
        if v_id not in self.nodes:
            raise ValueError(f"Nó '{v_id}' não existe. Adicione-o primeiro.")

        # Cria a conexão nos dois sentidos
        edge_uv = {'node': v_id, 'weight': weight, **kwargs}
        edge_vu = {'node': u_id, 'weight': weight, **kwargs}
        
        self.adj[u_id].append(edge_uv)
        self.adj[v_id].append(edge_vu)

    def get_neighbors(self, node_id) -> List[Dict[str, Any]]:
        """
        Retorna a lista de vizinhos de um nó.

        :param node_id: O ID do nó.
        :return: Uma lista de vizinhos (ou lista vazia se não tiver).
        """
        return self.adj.get(node_id, [])

    def get_node_attributes(self, node_id):
        """
        Retorna os atributos de um nó (tipo, nome, etc).

        :param node_id: O ID do nó.
        :return: Um dicionário com os atributos.
        """
        return self.nodes.get(node_id)

    def __str__(self):
        """
        Define como o grafo será exibido ao usar `print(g)`.
        """
        num_users = sum(1 for n in self.nodes if n.startswith('U'))
        num_movies = sum(1 for n in self.nodes if n.startswith('F'))
        num_genres = sum(1 for n in self.nodes if n.startswith('G'))
        num_edges = sum(len(edges) for edges in self.adj.values()) // 2  # Dividido por 2 porque é não-direcionado
        
        return (
            f"Grafo RacoGraph:\n"
            f"  Nós: {len(self.nodes)} total\n"
            f"    - Usuários: {num_users}\n"
            f"    - Filmes: {num_movies}\n"
            f"    - Gêneros: {num_genres}\n"
            f"  Arestas: {num_edges}\n"
        )