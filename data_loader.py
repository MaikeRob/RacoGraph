"""
Módulo responsável pelo carregamento de dados e construção do grafo.
Centraliza toda a lógica de leitura do MovieLens e criação da estrutura.
"""
from typing import Tuple
import pandas as pd

from graph import Graph
from constants import (
    MOVIES_FILE, 
    RATINGS_FILE,
    NODE_PREFIX_USER,
    NODE_PREFIX_MOVIE,
    NODE_PREFIX_GENRE,
    NODE_TYPE_USER,
    NODE_TYPE_MOVIE,
    NODE_TYPE_GENRE,
    RELATION_RATED,
    RELATION_BELONGS_TO_GENRE,
    NO_GENRE_LABEL
)


def load_movies() -> pd.DataFrame:
    """
    Carrega o arquivo de filmes do MovieLens.
    
    Returns:
        DataFrame com colunas: movieId, title, genres
    """
    return pd.read_csv(MOVIES_FILE, dtype={'movieId': int})


def load_ratings() -> pd.DataFrame:
    """
    Carrega o arquivo de avaliações do MovieLens.
    
    Returns:
        DataFrame com colunas: userId, movieId, rating, timestamp
    """
    return pd.read_csv(RATINGS_FILE, dtype={'userId': int, 'movieId': int})


def extract_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai todos os gêneros únicos dos filmes e cria DataFrame com IDs.
    
    Args:
        movies_df: DataFrame de filmes
        
    Returns:
        DataFrame com colunas: genreId, genre
    """
    genres = set()
    for genre_list in movies_df['genres']:
        for genre in str(genre_list).split('|'):
            if genre and genre != NO_GENRE_LABEL:
                genres.add(genre)
    
    genres_df = pd.DataFrame(sorted(genres), columns=['genre'])
    genres_df = genres_df.reset_index()
    genres_df = genres_df.rename(columns={'index': 'genreId'})
    
    return genres_df


def build_graph(movies_df: pd.DataFrame = None, 
                ratings_df: pd.DataFrame = None) -> Tuple[Graph, pd.DataFrame, pd.DataFrame]:
    """
    Constrói o grafo de recomendação completo a partir dos dados do MovieLens.
    
    Se movies_df e ratings_df não forem fornecidos, carrega os dados padrão.
    Isso permite usar a mesma função para treino/teste no eval.py.
    
    Args:
        movies_df: DataFrame de filmes (opcional)
        ratings_df: DataFrame de avaliações (opcional)
    
    Returns:
        Tuple contendo:
        - Graph: Grafo com nós (usuários, filmes, gêneros) e arestas
        - pd.DataFrame: DataFrame de filmes usado
        - pd.DataFrame: DataFrame de avaliações usado
    """
    # Carrega dados se não fornecidos
    if movies_df is None:
        movies_df = load_movies()
    if ratings_df is None:
        ratings_df = load_ratings()
    
    # Extrai gêneros
    genres_df = extract_genres(movies_df)
    
    # Cria instância do grafo
    graph = Graph()
    
    # Adiciona nós de gêneros
    for _, row in genres_df.iterrows():
        graph.add_node(
            node_id=f"{NODE_PREFIX_GENRE}{row['genreId']}",
            node_type=NODE_TYPE_GENRE,
            name=row['genre']
        )
    
    # Adiciona nós de filmes e suas relações com gêneros
    for _, row in movies_df.iterrows():
        movie_id = f"{NODE_PREFIX_MOVIE}{row['movieId']}"
        graph.add_node(
            node_id=movie_id,
            node_type=NODE_TYPE_MOVIE,
            title=row['title']
        )
        
        # Conecta filme aos seus gêneros
        for genre in str(row['genres']).split('|'):
            if genre == NO_GENRE_LABEL or not genre:
                continue
            
            genre_rows = genres_df[genres_df['genre'] == genre]
            if len(genre_rows) > 0:
                genre_id = f"{NODE_PREFIX_GENRE}{int(genre_rows.iloc[0]['genreId'])}"
                graph.add_edge(
                    u_id=movie_id,
                    v_id=genre_id,
                    weight=1.0,
                    relation_type=RELATION_BELONGS_TO_GENRE
                )
    
    # Adiciona nós de usuários e suas avaliações
    for _, row in ratings_df.iterrows():
        user_id = f"{NODE_PREFIX_USER}{int(row['userId'])}"
        movie_id = f"{NODE_PREFIX_MOVIE}{int(row['movieId'])}"
        
        # Cria o nó do usuário se ainda não existir
        if not graph.get_node(user_id):
            graph.add_node(
                node_id=user_id,
                node_type=NODE_TYPE_USER
            )
        
        # Cria aresta entre usuário e filme com o peso da avaliação
        graph.add_edge(
            u_id=user_id,
            v_id=movie_id,
            weight=float(row['rating']),
            relation_type=RELATION_RATED
        )
    
    return graph, movies_df, ratings_df
