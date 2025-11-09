"""
Sistema de recomendação baseado em Random Walk com Reinício (Personalized PageRank).

Implementa caminhadas aleatórias no grafo para descobrir filmes relevantes,
considerando a estrutura completa de conexões entre usuários, filmes e gêneros.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

from graph import is_user, is_movie
from constants import (
    DEFAULT_RESTART_PROB_USER,
    DEFAULT_RESTART_PROB_SIMILAR,
    DEFAULT_NUM_WALKS,
    DEFAULT_WALK_LENGTH,
    DEFAULT_MIN_USER_RATING,
    DEFAULT_TOP_N,
    DEFAULT_K_SIMILAR
)


# ============ Utilitários de Acesso ao Grafo ============

def get_user_movies(g, user_id: str) -> Dict[str, float]:
    """
    Retorna os filmes avaliados por um usuário com suas notas.
    Acessa o grafo diretamente, sem estruturas auxiliares.
    
    Args:
        g: Grafo
        user_id: ID do usuário (ex: "U1")
    
    Returns:
        Dicionário {movie_id: rating}
    """
    movies_ratings = {}
    
    for edge in g.get_neighbors(user_id):
        neighbor = edge.get("node")
        if is_movie(neighbor):
            rating = float(edge.get("weight", 1.0))
            movies_ratings[neighbor] = rating
    
    return movies_ratings


# ============ Random Walk com Reinício (Personalized PageRank) ============

def random_walk(
    g,
    start_nodes: List[str],
    num_walks: int = DEFAULT_NUM_WALKS,
    walk_length: int = DEFAULT_WALK_LENGTH,
    restart_prob: float = DEFAULT_RESTART_PROB_USER,
    weights: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Executa Random Walk com Reinício a partir de um conjunto de nós iniciais.
    
    Args:
        g: Grafo
        start_nodes: Lista de nós iniciais (ex: filmes avaliados pelo usuário)
        num_walks: Número de caminhadas a realizar
        walk_length: Comprimento máximo de cada caminhada
        restart_prob: Probabilidade de reiniciar em um nó inicial (tipicamente 0.15)
        weights: Pesos dos nós iniciais (ex: ratings do usuário)
    
    Returns:
        Dicionário com scores de visita para cada nó
    """
    if not start_nodes:
        return {}
    
    # Inicializa pesos uniformes se não fornecidos
    if weights is None:
        weights = {node: 1.0 for node in start_nodes}
    
    # Normaliza pesos para probabilidades
    total_weight = sum(weights.values())
    start_probs = {node: weights[node] / total_weight for node in start_nodes}
    
    # Contador de visitas
    visit_counts: Dict[str, float] = defaultdict(float)
    
    for _ in range(num_walks):
        # Escolhe nó inicial baseado nos pesos
        current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
        
        for step in range(walk_length):
            # Registra visita
            visit_counts[current] += 1.0
            
            # Decide se reinicia
            if random.random() < restart_prob:
                current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
                continue
            
            # Pega vizinhos
            neighbors = g.get_neighbors(current)
            if not neighbors:
                # Sem vizinhos, reinicia
                current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
                continue
            
            # Escolhe próximo nó ponderado pelo peso da aresta
            next_nodes = []
            edge_weights = []
            
            for edge in neighbors:
                next_node = edge.get("node")
                edge_weight = float(edge.get("weight", 1.0))
                next_nodes.append(next_node)
                edge_weights.append(edge_weight)
            
            # Normaliza pesos
            total_edge_weight = sum(edge_weights)
            if total_edge_weight > 0:
                edge_probs = [w / total_edge_weight for w in edge_weights]
                current = random.choices(next_nodes, weights=edge_probs)[0]
            else:
                current = random.choice(next_nodes)
    
    # Normaliza scores
    total_visits = sum(visit_counts.values())
    if total_visits > 0:
        visit_counts = {node: count / total_visits for node, count in visit_counts.items()}
    
    return visit_counts


def topk_similar_movies(
    g,
    movie_id: str,
    k: int = DEFAULT_TOP_N,
    metric: str = "randomwalk",
    min_co: int = 3,
    num_walks: int = 500,
    walk_length: int = 8,
) -> List[Tuple[str, float]]:
    """
    Retorna os K filmes mais similares a `movie_id` usando Random Walk.
    
    Args:
        g: Grafo
        movie_id: ID do filme de referência
        k: Número de filmes similares a retornar
        metric: Métrica a usar ("randomwalk" é o padrão)
        min_co: Mínimo de usuários em comum (usado para filtrar candidatos iniciais)
        num_walks: Número de caminhadas
        walk_length: Comprimento de cada caminhada
    
    Returns:
        Lista de tuplas (movie_id, score) ordenada por score decrescente
    """
    if movie_id not in g.nodes:
        return []
    
    # Executa random walk a partir do filme
    scores = random_walk(
        g,
        start_nodes=[movie_id],
        num_walks=num_walks,
        walk_length=walk_length,
        restart_prob=DEFAULT_RESTART_PROB_SIMILAR
    )
    
    # Filtra apenas filmes (exceto o próprio)
    movie_scores = [
        (node, score) 
        for node, score in scores.items() 
        if is_movie(node) and node != movie_id
    ]
    
    # Ordena por score
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    return movie_scores[:k]


def recommend_for_user(
    g,
    user_id: str,
    k_similar: int = DEFAULT_K_SIMILAR,
    topn: int = DEFAULT_TOP_N,
    metric: str = "randomwalk",
    min_co: int = 3,
    min_user_rating: float = DEFAULT_MIN_USER_RATING,
    num_walks: int = 2000,
    walk_length: int = 10,
) -> List[Tuple[str, float]]:
    """
    Gera recomendações para um usuário usando Random Walk com Reinício.
    
    O algoritmo:
    1. Identifica filmes avaliados pelo usuário (com nota >= min_user_rating)
    2. Executa random walks a partir desses filmes, ponderados pelas notas
    3. Conta visitas em nós do grafo (usuários, filmes, gêneros)
    4. Filtra apenas filmes não vistos
    5. Retorna top-N com maiores scores
    
    Args:
        g: Grafo
        user_id: ID do usuário
        k_similar: (não usado no random walk, mantido para compatibilidade)
        topn: Número de recomendações a retornar
        metric: Métrica a usar ("randomwalk" é o padrão)
        min_co: (não usado no random walk, mantido para compatibilidade)
        min_user_rating: Nota mínima para considerar um filme como preferido
        num_walks: Número de caminhadas aleatórias
        walk_length: Comprimento de cada caminhada
    
    Returns:
        Lista de tuplas (movie_id, score) ordenada por score decrescente
    """
    # Obtém filmes avaliados pelo usuário diretamente do grafo
    user_movies = get_user_movies(g, user_id)
    
    if not user_movies:
        return []
    
    # Filtra filmes bem avaliados pelo usuário
    start_movies = []
    movie_weights = {}
    
    for movie, rating in user_movies.items():
        if rating >= min_user_rating:
            start_movies.append(movie)
            movie_weights[movie] = rating
    
    if not start_movies:
        # Se nenhum filme passa o threshold, usa todos
        start_movies = list(user_movies.keys())
        movie_weights = user_movies.copy()
    
    # Executa random walk a partir dos filmes do usuário
    scores = random_walk(
        g,
        start_nodes=start_movies,
        num_walks=num_walks,
        walk_length=walk_length,
        restart_prob=DEFAULT_RESTART_PROB_USER,
        weights=movie_weights
    )
    
    # Filtra apenas filmes não vistos
    recommendations = [
        (node, score)
        for node, score in scores.items()
        if is_movie(node) and node not in user_movies
    ]
    
    # Ordena por score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations[:topn]
