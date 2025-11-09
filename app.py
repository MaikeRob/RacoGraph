"""
Interface web interativa do RacoGraph usando Streamlit.

Permite explorar o sistema de recomendação através de dois modos:
1. Encontrar filmes similares a um filme específico
2. Gerar recomendações personalizadas para um usuário
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data_loader import build_graph
from recommender import topk_similar_movies, recommend_for_user, get_user_movies
from constants import NODE_PREFIX_USER, NODE_PREFIX_MOVIE

# ========= CACHE =========
@st.cache_data(show_spinner=True)
def load_data():
    g, movies_df, ratings_df = build_graph()
    return g, movies_df, ratings_df

@st.cache_data
def build_title_maps(movies_df: pd.DataFrame):
    id_to_title = {int(r.movieId): r.title for r in movies_df.itertuples(index=False)}
    title_to_id = {v: k for k, v in id_to_title.items()}
    return id_to_title, title_to_id

def get_title(g, movies_df: pd.DataFrame, movie_node_id: str) -> str:
    """
    Obtém o título de um filme de forma segura.
    
    Tenta primeiro pelo DataFrame (mais rápido), depois pelo grafo, e por
    último retorna o próprio ID se nada funcionar.
    
    Args:
        g: Grafo contendo os nós
        movies_df: DataFrame com dados dos filmes
        movie_node_id: ID do nó do filme (formato "F123")
    
    Returns:
        Título do filme ou ID se não encontrado
    """
    try:
        mid = int(movie_node_id[1:])
        return movies_df.loc[movies_df["movieId"] == mid, "title"].iloc[0]
    except Exception:
        node = g.get_node(movie_node_id)
        return (node.get("title") if node else None) or movie_node_id

# ========= UI =========
st.set_page_config(page_title="RacoGraph – Recomendador por Grafos", layout="wide")
st.title("RacoGraph - Recomendacao por Grafos (MovieLens Small)")

with st.spinner("Carregando grafo e dados..."):
    g, movies_df, ratings_df = load_data()
    id_to_title, title_to_id = build_title_maps(movies_df)

st.sidebar.header("Parametros")
mode = st.sidebar.radio("Modo", ["Recomendar para usuário", "Filmes similares"])

# ==== FILTRO DE GÊNERO ====
all_genres = sorted(
    {g for gs in movies_df["genres"] for g in str(gs).split("|") if g and g != "(no genres listed)"}
)
selected_genre = st.sidebar.selectbox(
    "Filtrar por genero (opcional)",
    ["Todos"] + all_genres,
    index=0
)

# Parâmetros do Random Walk
st.sidebar.subheader("Random Walk")
num_walks = st.sidebar.slider(
    "Numero de caminhadas",
    500, 10000, 1000, 500,
    help="Mais caminhadas = mais precisão, mas mais lento"
)
walk_length = st.sidebar.slider(
    "Comprimento da caminhada",
    5, 20, 10, 1,
    help="Número de passos em cada caminhada aleatória"
)

# Parâmetros gerais
st.sidebar.subheader("Recomendacoes")
topk = st.sidebar.slider(
    "Quantidade de recomendacoes",
    5, 50, 10, 1,
    help="Número de filmes exibidos na lista final."
)
min_user_rating = st.sidebar.slider(
    "Nota minima do usuario",
    0.0, 5.0, 3.0, 0.5,
    help="Filmes avaliados abaixo disso não entram no cálculo do usuário."
)

colL, colR = st.columns([2, 3])

with colL:
    st.subheader("Resumo do Grafo")
    n_users = sum(1 for nid in g.nodes if str(nid).startswith(NODE_PREFIX_USER))
    n_movies = sum(1 for nid in g.nodes if str(nid).startswith(NODE_PREFIX_MOVIE))
    n_genres = sum(1 for nid in g.nodes if str(nid).startswith("G"))
    n_edges = sum(len(v) for v in g.adj.values()) // 2
    st.metric("Usuarios (U)", n_users)
    st.metric("Filmes (F)", n_movies)
    st.metric("Generos (G)", n_genres)
    st.metric("Arestas", n_edges)

with colR:
    st.subheader("Selecao")

    if mode == "Filmes similares":
        title = st.selectbox("Escolha um filme", sorted(title_to_id.keys()))
        movie_id = title_to_id[title]
        mid_node = f"{NODE_PREFIX_MOVIE}{int(movie_id)}"
        run = st.button("Buscar similares")

        if run:
            sims = topk_similar_movies(
                g, mid_node, k=topk, metric="randomwalk", 
                num_walks=num_walks, walk_length=walk_length
            )
            if not sims:
                st.warning("Nenhum similar encontrado com os parâmetros atuais.")
            else:
                df = pd.DataFrame(
                    [{"movieId": m, "title": get_title(g, movies_df, m), "similarity": s} for m, s in sims]
                )
                st.write(f"**Top {len(df)} similares a:** `{mid_node}` — *{title}*")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("title")["similarity"])

    else:
        user_id = st.selectbox("Escolha um usuario", sorted(ratings_df["userId"].unique().tolist()))
        uid_node = f"{NODE_PREFIX_USER}{int(user_id)}"
        run = st.button("Recomendar")

        if run:
            # Mostra filmes que o usuário gostou (usados para caminhada)
            user_movies = get_user_movies(g, uid_node)
            liked_movies = {mid: rating for mid, rating in user_movies.items() 
                          if rating >= min_user_rating}
            
            if liked_movies:
                st.subheader(f"Filmes que o usuario {user_id} gostou (nota >= {min_user_rating})")
                st.caption(f"Estes {len(liked_movies)} filmes serao usados como ponto de partida para o Random Walk")
                
                liked_df = pd.DataFrame([
                    {
                        "movieId": mid,
                        "title": get_title(g, movies_df, mid),
                        "rating": rating
                    }
                    for mid, rating in sorted(liked_movies.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(liked_df, use_container_width=True)
                st.divider()
            else:
                st.warning(f"Usuario {user_id} nao tem filmes avaliados com nota >= {min_user_rating}")
            
            # Gera recomendações
            st.subheader("Recomendacoes Personalizadas")
            recs = recommend_for_user(
                g, uid_node,
                k_similar=30, topn=topk,
                metric="randomwalk", min_user_rating=min_user_rating,
                num_walks=num_walks, walk_length=walk_length
            )

            # aplica filtro por gênero se selecionado
            if selected_genre != "Todos":
                allowed_ids = set(
                    movies_df[movies_df["genres"].str.contains(selected_genre, case=False, na=False)]["movieId"].tolist()
                )
                recs = [(mid, score) for mid, score in recs if int(mid[1:]) in allowed_ids]

            if not recs:
                st.warning("Sem recomendações por similaridade. Mostrando populares como fallback.")
                if selected_genre != "Todos":
                    allowed_ids = set(
                        movies_df[movies_df["genres"].str.contains(selected_genre, case=False, na=False)]["movieId"].tolist()
                    )
                    popular_ids = (
                        ratings_df[ratings_df["movieId"].isin(allowed_ids)]["movieId"]
                        .value_counts()
                        .head(topk)
                        .index.tolist()
                    )
                else:
                    popular_ids = ratings_df["movieId"].value_counts().head(topk).index.tolist()

                df = pd.DataFrame(
                    [{"movieId": f"{NODE_PREFIX_MOVIE}{int(x)}", "title": get_title(g, movies_df, f"{NODE_PREFIX_MOVIE}{int(x)}"), "score": None}
                     for x in popular_ids]
                )
            else:
                df = pd.DataFrame(
                    [{"movieId": m, "title": get_title(g, movies_df, m), "score": s} for m, s in recs]
                )

            st.write(f"**Recomendacoes para:** `{uid_node}`"
                     + (f" - filtro: *{selected_genre}*" if selected_genre != "Todos" else ""))
            st.dataframe(df, use_container_width=True)
            if "score" in df and df["score"].notna().any():
                st.bar_chart(df.set_index("title")["score"])
