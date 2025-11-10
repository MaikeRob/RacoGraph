"""Estrutura de grafo não-direcionado usando lista de adjacência."""
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
        """Inicializa grafo vazio com nós e adjacências."""
        self.nodes = {}
        self.adj = {}

    def get_node(self, node_id):
        """Retorna os atributos de um nó ou None."""
        return self.nodes.get(node_id)


    def add_node(self, node_id, node_type, **kwargs):
        """Adiciona um nó ao grafo."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {'type': node_type, **kwargs}
            self.adj[node_id] = []

    def add_edge(self, u_id, v_id, weight=1.0, **kwargs):
        """Adiciona aresta não-direcionada entre dois nós."""
        if u_id not in self.nodes:
            raise ValueError(f"Nó '{u_id}' não existe. Adicione-o primeiro.")
        if v_id not in self.nodes:
            raise ValueError(f"Nó '{v_id}' não existe. Adicione-o primeiro.")

        self.adj[u_id].append({'node': v_id, 'weight': weight, **kwargs})
        self.adj[v_id].append({'node': u_id, 'weight': weight, **kwargs})

    def get_neighbors(self, node_id) -> List[Dict[str, Any]]:
        """Retorna lista de vizinhos do nó."""
        return self.adj.get(node_id, [])

    def __str__(self):
        """Representação string do grafo."""
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