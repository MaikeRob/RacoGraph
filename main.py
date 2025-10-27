# main.py

import argparse
import sys
import os
from datetime import datetime
import pandas as pd
from graph import Graph
from recommender import topk_similar_movies, recommend_for_user


def build_graph():
    # Carrega os dados de filmes e avaliações do MovieLens
    movies_data = pd.read_csv('data/ml-latest-small/movies.csv', dtype={'movieId': int})
    ratings_data = pd.read_csv('data/ml-latest-small/ratings.csv', dtype={'userId': int, 'movieId': int})

    # Extrai todos os gêneros únicos dos filmes
    genres = set()
    for genre_list in movies_data['genres']:
        for genre in genre_list.split('|'):
            genres.add(genre)

    genres = list(genres)
    # Remove a categoria vazia, se existir
    if '(no genres listed)' in genres:
        genres.remove('(no genres listed)')

    # Cria DataFrame de gêneros com IDs
    genres = pd.DataFrame(genres, columns=['genre'])
    genres = genres.reset_index()
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

    return graph_instance, movies_data, ratings_data


def main():
    parser = argparse.ArgumentParser(
        description="RacoGraph – construir grafo e gerar recomendações (Item–Item ou User-based)."
    )
    parser.add_argument("--movie-id", type=int, help="Mostrar filmes similares a este movieId (ex: 1)")
    parser.add_argument("--user-id", type=int, help="Gerar recomendações para este userId (ex: 1)")
    parser.add_argument("--metric", choices=["jaccard", "cosine"], default="jaccard",
                        help="Métrica de similaridade (padrão: jaccard)")
    parser.add_argument("--min-co", type=int, default=3, help="Mínimo de usuários em comum (padrão: 3)")
    parser.add_argument("--topk", type=int, default=10, help="Quantidade de itens a listar (padrão: 10)")
    parser.add_argument("--no-summary", action="store_true", help="Não imprimir resumo do grafo")
    args = parser.parse_args()

    g, movies_df, ratings_df = build_graph()

    if not args.no_summary:
        print(g)

    # ==============================
    # Similares a um filme específico
    # ==============================
    if args.movie_id is not None:
        mid = f"F{int(args.movie_id)}"
        sims = topk_similar_movies(g, mid, k=args.topk, metric=args.metric, min_co=args.min_co)
        title_ref = movies_df.loc[movies_df["movieId"] == args.movie_id, "title"].iloc[0] \
            if (movies_df["movieId"] == args.movie_id).any() else mid
        print(f"\nTop {args.topk} similares a {mid} ({title_ref}):")
        if not sims:
            print("Nenhum similar encontrado. Tente reduzir --min-co (ex.: --min-co 1) ou usar --metric jaccard.")
        for m, s in sims:
            try:
                m_int = int(m[1:])
                title = movies_df.loc[movies_df["movieId"] == m_int, "title"].iloc[0]
            except Exception:
                title = g.get_node(m).get("title", m)
            print(f"{m:<8} {s:.4f} | {title}")

        if args.user_id is None:
            return

    # ==============================
    # Recomendações para um usuário
    # ==============================
    recs = []
    if args.user_id is not None:
        uid = f"U{int(args.user_id)}"
        recs = recommend_for_user(
            g, uid, k_similar=30, topn=args.topk,
            metric=args.metric, min_co=args.min_co, min_user_rating=3.5
        )
        print(f"\nRecomendações para {uid}:")
        if not recs:
            print("Sem recomendações. Tente diminuir --min-co (ex.: 1) ou usar --metric jaccard.")
        for m, score in recs:
            try:
                m_int = int(m[1:])
                title = movies_df.loc[movies_df["movieId"] == m_int, "title"].iloc[0]
            except Exception:
                title = g.get_node(m).get("title", m)
            print(f"{m:<8} {score:.4f} | {title}")

    # ==============================
    # Salva recomendações em CSV
    # ==============================
    if args.user_id is not None and recs:
        os.makedirs("outputs", exist_ok=True)
        csv_path = os.path.join("outputs", f"recommendations_user{args.user_id}.csv")

        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId", "title", "score", "metric", "timestamp"])
            for mid, score in recs:
                try:
                    mid_int = int(mid[1:])
                    title = movies_df.loc[movies_df["movieId"] == mid_int, "title"].iloc[0]
                except Exception:
                    title = g.get_node(mid).get("title", mid)
                writer.writerow([
                    args.user_id,
                    mid,
                    title,
                    f"{score:.4f}",
                    args.metric,
                    datetime.now().isoformat(timespec="seconds")
                ])

        print(f"\n📁 Recomendações salvas em: {csv_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
