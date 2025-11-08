# eval.py
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

from graph import Graph
from recommender import recommend_for_user

DATA_DIR = Path("data/ml-latest-small")

# -------------------- Split de treino/teste --------------------

def split_per_user(ratings: pd.DataFrame, mode: str = "last", holdout: int = 1, test_frac: float = 0.2, seed: int = 42):
    """
    Retorna (train_df, test_df) por usuário.
    mode:
      - "last": usa as últimas 'holdout' avaliações (por timestamp) de cada usuário no teste
      - "random": amostra 'test_frac' das avaliações no teste (por usuário)
    """
    assert mode in {"last", "random"}
    rng = np.random.default_rng(seed)

    if "timestamp" in ratings.columns:
        ratings = ratings.sort_values(["userId", "timestamp"])
    else:
        ratings = ratings.sort_values(["userId"])  # fallback

    train_parts = []
    test_parts = []

    for uid, grp in ratings.groupby("userId"):
        n = len(grp)
        if n <= 1:
            # sem material para split, joga tudo em treino
            train_parts.append(grp)
            continue

        if mode == "last":
            t = grp.tail(holdout)
            tr = grp.iloc[:-holdout]
        else:  # random
            m = max(1, int(math.ceil(n * test_frac)))
            idx = grp.index.to_list()
            test_idx = set(rng.choice(idx, size=m, replace=False).tolist())
            t = grp.loc[list(test_idx)]
            tr = grp.drop(test_idx)

        if len(tr) == 0:
            # garante pelo menos 1 no treino
            tr = grp.iloc[:-1]
            t = grp.iloc[-1:]

        train_parts.append(tr)
        test_parts.append(t)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0]
    return train_df, test_df

# -------------------- Construção do grafo (igual ao seu main) --------------------

def build_graph(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Graph:
    # gêneros
    genres = set()
    for genre_list in movies_df["genres"]:
        for g in str(genre_list).split("|"):
            genres.add(g)
    genres = [g for g in genres if g and g != "(no genres listed)"]

    genres_df = pd.DataFrame(genres, columns=["genre"]).reset_index().rename(columns={"index": "genreId"})

    g = Graph()

    # nós de gênero
    for _, row in genres_df.iterrows():
        g.add_node(node_id=f"G{row['genreId']}", node_type="Genre", name=row["genre"])

    # filmes e arestas filme–gênero
    for _, row in movies_df.iterrows():
        fid = f"F{int(row['movieId'])}"
        g.add_node(node_id=fid, node_type="Movie", title=row["title"])
        for gen in str(row["genres"]).split("|"):
            if gen == "(no genres listed)":
                continue
            gid = genres_df.loc[genres_df["genre"] == gen, "genreId"]
            if len(gid):
                g.add_edge(u_id=fid, v_id=f"G{int(gid.iloc[0])}", weight=1.0, relation_type="pertence_ao_genero")

    # usuários e avaliações (APENAS do conjunto de treino!)
    for _, row in ratings_df.iterrows():
        uid = f"U{int(row['userId'])}"
        fid = f"F{int(row['movieId'])}"
        if not g.get_node(uid):
            g.add_node(node_id=uid, node_type="User")
        g.add_edge(u_id=uid, v_id=fid, weight=float(row["rating"]), relation_type="avaliou")

    return g

# -------------------- Métricas @K --------------------

def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k == 0: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / k

def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant: return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for i in rec_k if i in relevant)
    return hits / len(relevant)

def apk(recommended: list[str], relevant: set[str], k: int) -> float:
    """Average Precision @K (AP de um usuário)."""
    if not relevant: return 0.0
    score = 0.0
    hits = 0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / min(len(relevant), k)

def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Ganho com desconto binário."""
    def dcg(items):
        s = 0.0
        for i, it in enumerate(items[:k], start=1):
            rel = 1.0 if it in relevant else 0.0
            if rel:
                s += rel / math.log2(i + 1)
        return s
    ideal = dcg(list(relevant))  # ordem ideal: todos relevantes no topo
    if ideal == 0.0:
        return 0.0
    return dcg(recommended) / ideal

# -------------------- Avaliação --------------------

def evaluate(g: Graph,
             movies_df: pd.DataFrame,
             train_df: pd.DataFrame,
             test_df: pd.DataFrame,
             k: int = 10,
             metric: str = "jaccard",
             min_co: int = 3,
             min_user_rating: float = 3.5):
    users = sorted(test_df["userId"].unique().tolist())

    precs, recs, maps, ndcgs, hits = [], [], [], [], []
    all_recommended = set()
    evaluated_users = 0

    for uid_int in users:
        uid = f"U{uid_int}"
        # relevantes = filmes no TESTE do usuário
        rel_movies = set(f"F{int(m)}" for m in test_df.loc[test_df["userId"] == uid_int, "movieId"].tolist())
        if not rel_movies:
            continue

        # recomendações a partir do TREINO
        recs_user = recommend_for_user(
            g, uid,
            k_similar=30, topn=k, metric=metric, min_co=min_co, min_user_rating=min_user_rating
        )
        rec_list = [m for (m, score) in recs_user]

        if rec_list:
            all_recommended.update(rec_list)

        p = precision_at_k(rec_list, rel_movies, k)
        r = recall_at_k(rec_list, rel_movies, k)
        ap = apk(rec_list, rel_movies, k)
        nd = ndcg_at_k(rec_list, rel_movies, k)
        hit = 1.0 if any(m in rel_movies for m in rec_list[:k]) else 0.0

        precs.append(p); recs.append(r); maps.append(ap); ndcgs.append(nd); hits.append(hit)
        evaluated_users += 1

    results = {
        "users_evaluated": evaluated_users,
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"Recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        "Coverage": len(all_recommended) / movies_df["movieId"].nunique()
    }
    return results

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Avaliação Top-N do RacoGraph (MovieLens Small).")
    parser.add_argument("--k", type=int, default=10, help="Top-K para métricas (default: 10)")
    parser.add_argument("--metric", choices=["jaccard", "cosine"], default="jaccard", help="Métrica de similaridade")
    parser.add_argument("--min-co", type=int, default=3, help="Mínimo de coavaliações para considerar par de filmes")
    parser.add_argument("--split", choices=["last", "random"], default="last", help="Como separar treino/teste por usuário")
    parser.add_argument("--holdout", type=int, default=1, help="Se split=last, nº de itens no teste por usuário")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Se split=random, fração por usuário para teste")
    parser.add_argument("--min-user-rating", type=float, default=3.5, help="Nota mínima para o item pesar na recomendação")
    args = parser.parse_args()

    # carrega dados
    movies = pd.read_csv(DATA_DIR / "movies.csv", dtype={"movieId": int})
    ratings = pd.read_csv(DATA_DIR / "ratings.csv", dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})

    # split
    train_df, test_df = split_per_user(
        ratings, mode=args.split, holdout=args.holdout, test_frac=args.test_frac
    )

    # grafo com APENAS TREINO
    g = build_graph(movies, train_df)

    # avaliação
    res = evaluate(
        g, movies, train_df, test_df,
        k=args.k, metric=args.metric, min_co=args.min_co, min_user_rating=args.min_user_rating
    )

    print("\n=== RESULTADOS ===")
    for k_, v in res.items():
        if isinstance(v, float):
            print(f"{k_:<14}: {v:.4f}")
        else:
            print(f"{k_:<14}: {v}")

if __name__ == "__main__":
    main()
