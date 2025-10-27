# recommender.py
from __future__ import annotations
from collections import defaultdict
from math import sqrt
from typing import Dict, Tuple, List, Set, Any

# IDs no seu projeto:
#  - Usuário: "U<userId>"
#  - Filme:   "F<movieId>"
#  - Gênero:  "G<genreId>"

def is_user(nid: str) -> bool:
    return isinstance(nid, str) and nid.startswith("U")

def is_movie(nid: str) -> bool:
    return isinstance(nid, str) and nid.startswith("F")

def _neighbors(g, node_id: str) -> List[Dict[str, Any]]:
    """Acessa vizinhos do nó, usando get_neighbors se existir, senão g.adj."""
    if hasattr(g, "get_neighbors"):
        return g.get_neighbors(node_id) or []
    return g.adj.get(node_id, [])

def build_user_movie_maps(g) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[Tuple[str, str], float]]:
    """
    Constrói estruturas para recomendação:
      - users_movies[u]  = {F...}  (filmes avaliados pelo usuário u)
      - movies_users[f]  = {U...}  (usuários que avaliaram o filme f)
      - ratings[(u, f)]  = rating  (peso da aresta U–F; se não houver, usa 1.0)
    """
    users_movies: Dict[str, Set[str]] = defaultdict(set)
    movies_users: Dict[str, Set[str]] = defaultdict(set)
    ratings: Dict[Tuple[str, str], float] = {}

    for nid in g.nodes.keys():
        if not is_user(nid):
            continue
        for e in _neighbors(g, nid):
            v = e.get("node")
            if is_movie(v):
                users_movies[nid].add(v)
                movies_users[v].add(nid)
                ratings[(nid, v)] = float(e.get("weight", 1.0))

    return users_movies, movies_users, ratings

# ---------------- Similaridade item–item ----------------

def jaccard_users(f1: str, f2: str, movies_users: Dict[str, Set[str]]) -> float:
    """Jaccard entre dois filmes (conjuntos de usuários)."""
    u1, u2 = movies_users.get(f1, set()), movies_users.get(f2, set())
    if not u1 or not u2:
        return 0.0
    inter = len(u1 & u2)
    if inter == 0:
        return 0.0
    union = len(u1 | u2)
    return inter / union

def cosine_users(f1: str, f2: str,
                 movies_users: Dict[str, Set[str]],
                 ratings: Dict[Tuple[str, str], float]) -> float:
    """
    Cosseno entre vetores de usuários (dimensões = usuários), ponderado pelo rating.
    """
    u1, u2 = movies_users.get(f1, set()), movies_users.get(f2, set())
    common = u1 & u2
    if not common:
        return 0.0

    num = sum(ratings[(u, f1)] * ratings[(u, f2)] for u in common)
    den1 = sqrt(sum(ratings[(u, f1)] ** 2 for u in u1))
    den2 = sqrt(sum(ratings[(u, f2)] ** 2 for u in u2))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)

# ---------------- Recomendação ----------------

def topk_similar_movies(
    g,
    movie_id: str,
    k: int = 10,
    metric: str = "jaccard",   # "jaccard" ou "cosine"
    min_co: int = 3,           # mínimo de usuários em comum
) -> List[Tuple[str, float]]:
    """
    Retorna os K filmes mais similares a `movie_id`, considerando coavaliações (min_co).
    """
    _, movies_users, ratings = build_user_movie_maps(g)
    if movie_id not in movies_users:
        return []

    target_users = movies_users[movie_id]
    candidates: Set[str] = set()
    for u in target_users:
        for e in _neighbors(g, u):
            v = e.get("node")
            if is_movie(v) and v != movie_id:
                candidates.add(v)

    sims: List[Tuple[str, float]] = []
    for f2 in candidates:
        co = len(movies_users[movie_id] & movies_users[f2])
        if co < min_co:
            continue

        if metric == "cosine":
            s = cosine_users(movie_id, f2, movies_users, ratings)
        else:
            s = jaccard_users(movie_id, f2, movies_users)
        if s > 0:
            sims.append((f2, s))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

def recommend_for_user(
    g,
    user_id: str,
    k_similar: int = 20,
    topn: int = 10,
    metric: str = "jaccard",
    min_co: int = 3,
    min_user_rating: float = 3.5,
) -> List[Tuple[str, float]]:
    """
    Top-N para o usuário:
      score(candidato) = soma( sim(f, candidato) * rating(u, f) ) para f vistos por u com rating >= min_user_rating
    """
    users_movies, _, ratings = build_user_movie_maps(g)
    seen = users_movies.get(user_id, set())
    if not seen:
        return []

    cand_scores: Dict[str, float] = defaultdict(float)
    for f in seen:
        sims = topk_similar_movies(g, f, k=k_similar, metric=metric, min_co=min_co)
        r_uf = ratings.get((user_id, f), 0.0)
        if r_uf < min_user_rating:
            continue
        for c_id, s in sims:
            if c_id in seen:
                continue
            cand_scores[c_id] += s * r_uf

    ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:topn]
