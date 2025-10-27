# Importa bibliotecas necessárias
import pandas as pd
from graph import Graph

# Carrega os dados de filmes e avaliações do MovieLens
movies_data = pd.read_csv('data/ml-latest-small/movies.csv', dtype={'movieId': int})
ratings_data = pd.read_csv('data/ml-latest-small/ratings.csv', dtype={'userId': int, 'movieId': int})

# Extrai todos os gêneros únicos dos filmes
genres = set()
for genre_list in movies_data['genres']:
    for genre in genre_list.split('|'):
        genres.add(genre)
        
genres = list(genres)
# Remove a categoria vazia
genres.remove('(no genres listed)')


# Cria DataFrame de gêneros com IDs
# Cria DataFrame de gêneros com IDs
genres = pd.DataFrame(genres, columns=['genre'])

genres = genres.reset_index()

# Renomeia a coluna de índice para genreId
genres = genres.rename(columns={'index': 'genreId'})



# Cria instância do grafo
graph_instance = Graph()

# Adiciona os nós de gêneros ao grafo
for _, row in genres.iterrows():
    graph_instance.add_node(
        node_id=f"G{row['genreId']}",
        node_type='Genre',
        name=row['genre']
    )

# Adiciona os nós de filmes e suas relações com gêneros
for _, row in movies_data.iterrows():
    graph_instance.add_node(
        node_id=f"F{row['movieId']}",
        node_type='Movie',
        title=row['title']
    )
    # Cria arestas entre filmes e seus gêneros
    for genre in row['genres'].split('|'):
        if genre == '(no genres listed)':
            continue
        genre_id = genres[genres['genre'] == genre]['genreId'].values[0]
        graph_instance.add_edge(
            u_id=f"F{row['movieId']}",
            v_id=f"G{genre_id}",
            weight=1.0,
            relation_type='pertence_ao_genero'
        )

# Adiciona os nós de usuários e suas avaliações de filmes
for _, row in ratings_data.iterrows():
    user_node_id = f"U{int(row['userId'])}"
    movie_node_id = f"F{int(row['movieId'])}"

    # Cria o nó do usuário se ainda não existir
    if not graph_instance.get_node(user_node_id):
        graph_instance.add_node(
            node_id=user_node_id,
            node_type='User'
        )
        
    # Cria aresta entre usuário e filme com o peso da avaliação
    graph_instance.add_edge(
        u_id=user_node_id,
        v_id=movie_node_id,
        weight=row['rating'],
        relation_type='avaliou'
    )

# Imprime informações do grafo
print(graph_instance)