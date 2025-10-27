import json

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

    def get_neighbors(self, node_id):
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
        output = "Estrutura do Grafo:\n"
        for node, attrs in self.nodes.items():
            output += f"- Nó: {node} (Tipo: {attrs['type']})\n"
            neighbors = self.get_neighbors(node)
            if neighbors:
                for edge in neighbors:
                    output += f"  -> conecta com: {edge['node']} (Peso: {edge['weight']})\n"
            else:
                output += "  (sem conexões)\n"
        return output

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # 1. Criar o grafo
    g = Graph()

    # 2. Adicionar nós
    g.add_node('U1', node_type='User')
    g.add_node('U2', node_type='User')
    g.add_node('F1', node_type='Movie', title='Filme Exemplo 1')
    g.add_node('F2', node_type='Movie', title='Filme Exemplo 2')
    g.add_node('G1', node_type='Genre', name='Ação')
    g.add_node('G2', node_type='Genre', name='Comédia')

    # 3. Adicionar conexões (arestas)
    
    # U1 avaliou F1 com nota 5.0
    g.add_edge('U1', 'F1', weight=5.0, relation_type='avaliou')
    
    # U1 avaliou F2 com nota 3.0
    g.add_edge('U1', 'F2', weight=3.0, relation_type='avaliou')

    # F1 pertence ao gênero G1
    g.add_edge('F1', 'G1', weight=1.0, relation_type='pertence_ao_genero')
    
    # F2 pertence ao gênero G2
    g.add_edge('F2', 'G2', weight=1.0, relation_type='pertence_ao_genero')

    # 4. Imprimir o grafo
    print(g)

    # 5. Testar como pegar os vizinhos
    print("\n--- Teste de Vizinhos ---")
    usuario_teste = 'U1'
    print(f"Vizinhos de {usuario_teste}:")
    vizinhos_u1 = g.get_neighbors(usuario_teste)
    print(json.dumps(vizinhos_u1, indent=2))

    filme_teste = 'F1'
    print(f"\nVizinhos de {filme_teste}:")
    vizinhos_f1 = g.get_neighbors(filme_teste)
    print(json.dumps(vizinhos_f1, indent=2))