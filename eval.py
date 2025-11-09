"""
Módulo de avaliação do sistema de recomendação RacoGraph.

Implementa avaliação offline usando split treino/teste e métricas padrão:
- Precision@K: Proporção de recomendações relevantes
- Recall@K: Proporção de itens relevantes encontrados
- MAP@K: Mean Average Precision (qualidade do ranking)
- NDCG@K: Normalized Discounted Cumulative Gain
- HitRate@K: % de usuários com pelo menos 1 hit
- Coverage: Diversidade do catálogo recomendado

Uso:
    python eval.py --k 10 --num-walks 1000 --split last
"""
from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np
import pandas as pd

from graph import Graph
from recommender import recommend_for_user
from data_loader import build_graph
from constants import DATA_DIR

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
             num_walks: int = 1000,
             walk_length: int = 10,
             min_user_rating: float = 3.0) -> Dict:
    """
    Avalia o sistema de recomendação usando Random Walk.
    
    Args:
        g: Grafo construído com dados de TREINO apenas
        movies_df: DataFrame com informações dos filmes
        train_df: DataFrame de avaliações de treino
        test_df: DataFrame de avaliações de teste (ground truth)
        k: Número de recomendações a avaliar (top-K)
        num_walks: Número de caminhadas aleatórias
        walk_length: Comprimento de cada caminhada
        min_user_rating: Nota mínima para considerar preferência
    
    Returns:
        Dicionário com métricas de avaliação
    """
    users = sorted(test_df["userId"].unique().tolist())

    precs, recs, maps, ndcgs, hits = [], [], [], [], []
    all_recommended = set()
    evaluated_users = 0
    users_with_recs = 0

    print(f"\n🔄 Avaliando {len(users)} usuários...")
    print(f"📊 Parâmetros: k={k}, num_walks={num_walks}, walk_length={walk_length}, min_rating={min_user_rating}")
    
    for i, uid_int in enumerate(users, 1):
        if i % 50 == 0:
            print(f"   Progresso: {i}/{len(users)} usuários avaliados...")
        
        uid = f"U{uid_int}"
        
        # Filmes relevantes = filmes no TESTE do usuário
        rel_movies = set(f"F{int(m)}" for m in test_df.loc[test_df["userId"] == uid_int, "movieId"].tolist())
        if not rel_movies:
            continue

        # Gera recomendações usando APENAS dados de TREINO
        recs_user = recommend_for_user(
            g, uid,
            k_similar=30, 
            topn=k, 
            metric="randomwalk",
            min_user_rating=min_user_rating,
            num_walks=num_walks,
            walk_length=walk_length
        )
        rec_list = [m for (m, score) in recs_user]

        if rec_list:
            all_recommended.update(rec_list)
            users_with_recs += 1

        # Calcula métricas
        p = precision_at_k(rec_list, rel_movies, k)
        r = recall_at_k(rec_list, rel_movies, k)
        ap = apk(rec_list, rel_movies, k)
        nd = ndcg_at_k(rec_list, rel_movies, k)
        hit = 1.0 if any(m in rel_movies for m in rec_list[:k]) else 0.0

        precs.append(p)
        recs.append(r)
        maps.append(ap)
        ndcgs.append(nd)
        hits.append(hit)
        evaluated_users += 1

    total_movies = movies_df["movieId"].nunique()
    
    results = {
        "users_evaluated": evaluated_users,
        "users_with_recs": users_with_recs,
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"Recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        "Coverage": len(all_recommended) / total_movies if total_movies > 0 else 0.0,
        "unique_movies_recommended": len(all_recommended)
    }
    return results

# -------------------- CLI --------------------

def print_results(results: Dict, config: Dict):
    """Imprime resultados da avaliação de forma formatada."""
    print("\n" + "="*60)
    print("  RESULTADOS DA AVALIACAO - RacoGraph (Random Walk)")
    print("="*60)
    
    print("\n📋 Configuração:")
    for key, value in config.items():
        print(f"   {key:<20}: {value}")
    
    print("\n📊 Métricas de Qualidade:")
    
    # Separa métricas por tipo
    basic_metrics = ["users_evaluated", "users_with_recs", "unique_movies_recommended"]
    quality_metrics = [k for k in results.keys() if "@" in k]
    other_metrics = [k for k in results.keys() if k not in basic_metrics and "@" not in k]
    
    print("\n   Estatísticas Básicas:")
    for key in basic_metrics:
        if key in results:
            print(f"   {key:<30}: {results[key]}")
    
    print("\n   Métricas de Ranking:")
    for key in quality_metrics:
        value = results[key]
        print(f"   {key:<30}: {value:.4f}")
    
    print("\n   Outras Métricas:")
    for key in other_metrics:
        value = results[key]
        if isinstance(value, float):
            print(f"   {key:<30}: {value:.4f}")
        else:
            print(f"   {key:<30}: {value}")
    
    print("\n" + "="*60)
    
    # Interpretação
    print("\n💡 Interpretação:")
    map_val = results.get("MAP@10", 0)
    hitrate = results.get("HitRate@10", 0)
    coverage = results.get("Coverage", 0)
    
    if map_val > 0.15:
        print("   ✅ MAP: Excelente - modelo ranqueia bem itens relevantes")
    elif map_val > 0.10:
        print("   ⚠️  MAP: Bom - há espaço para melhorias no ranking")
    else:
        print("   ❌ MAP: Baixo - considere ajustar parâmetros")
    
    if hitrate > 0.70:
        print("   ✅ HitRate: Excelente - maioria dos usuários recebe recomendações úteis")
    elif hitrate > 0.50:
        print("   ⚠️  HitRate: Moderado - muitos usuários sem hits")
    else:
        print("   ❌ HitRate: Baixo - poucas recomendações relevantes")
    
    if coverage > 0.20:
        print("   ✅ Coverage: Boa diversidade no catálogo")
    elif coverage > 0.10:
        print("   ⚠️  Coverage: Moderada - sistema um pouco enviesado")
    else:
        print("   ❌ Coverage: Baixa - muito focado em poucos filmes")
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Avaliação offline do RacoGraph usando Random Walk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Avaliação básica (padrão)
  python eval.py
  
  # Avaliar top-20 com mais caminhadas
  python eval.py --k 20 --num-walks 5000
  
  # Split aleatório com 20% no teste
  python eval.py --split random --test-frac 0.2
  
  # Apenas filmes com nota >= 4.0
  python eval.py --min-user-rating 4.0
  
  # Avaliação completa otimizada
  python eval.py --k 10 --num-walks 2000 --walk-length 15 --min-user-rating 3.5
        """
    )
    
    # Parâmetros de avaliação
    parser.add_argument("--k", type=int, default=10, 
                       help="Top-K para métricas (default: 10)")
    
    # Parâmetros do Random Walk
    parser.add_argument("--num-walks", type=int, default=1000,
                       help="Numero de caminhadas aleatorias (default: 1000)")
    parser.add_argument("--walk-length", type=int, default=10,
                       help="Comprimento de cada caminhada (default: 10)")
    
    # Parâmetros de split
    parser.add_argument("--split", choices=["last", "random"], default="last",
                       help="Modo de split treino/teste (default: last)")
    parser.add_argument("--holdout", type=int, default=1,
                       help="Se split=last, numero de itens no teste por usuario (default: 1)")
    parser.add_argument("--test-frac", type=float, default=0.2,
                       help="Se split=random, fracao no teste (default: 0.2)")
    
    # Parâmetros do modelo
    parser.add_argument("--min-user-rating", type=float, default=3.0,
                       help="Nota minima para considerar preferencia (default: 3.0)")
    
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RacoGraph - Avaliacao de Sistema de Recomendacao")
    print("="*60)
    
    # Carrega dados
    print("\n📂 Carregando dados do MovieLens Small...")
    movies = pd.read_csv(DATA_DIR / "movies.csv", dtype={"movieId": int})
    ratings = pd.read_csv(DATA_DIR / "ratings.csv", 
                         dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    
    print(f"   ✓ {len(movies)} filmes")
    print(f"   ✓ {len(ratings)} avaliacoes")
    print(f"   ✓ {ratings['userId'].nunique()} usuarios")

    # Split treino/teste
    print(f"\n🔀 Dividindo dados (modo: {args.split})...")
    train_df, test_df = split_per_user(
        ratings, mode=args.split, holdout=args.holdout, test_frac=args.test_frac
    )
    print(f"   ✓ Treino: {len(train_df)} avaliacoes")
    print(f"   ✓ Teste: {len(test_df)} avaliacoes")

    # Constrói grafo com APENAS treino
    print("\n🔨 Construindo grafo com dados de TREINO...")
    g, _, _ = build_graph(movies, train_df)
    print(f"   ✓ Grafo construido com sucesso")
    print(f"   ✓ {len(g.nodes)} nos no grafo")

    # Configuração da avaliação
    config = {
        "Top-K": args.k,
        "Num Walks": args.num_walks,
        "Walk Length": args.walk_length,
        "Min User Rating": args.min_user_rating,
        "Split Mode": args.split,
        "Holdout" if args.split == "last" else "Test Frac": 
            args.holdout if args.split == "last" else args.test_frac
    }

    # Avaliação
    results = evaluate(
        g, movies, train_df, test_df,
        k=args.k, 
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        min_user_rating=args.min_user_rating
    )

    # Exibe resultados
    print_results(results, config)


if __name__ == "__main__":
    main()
