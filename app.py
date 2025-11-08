# app.py
from __future__ import annotations
import pandas as pd
import streamlit as st

# Importa do seu projeto
from main import build_graph
from recommender import topk_similar_movies, recommend_for_user

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
    try:
        mid = int(movie_node_id[1:])
        return movies_df.loc[movies_df["movieId"] == mid, "title"].iloc[0]
    except Exception:
        node = g.get_node(movie_node_id)
        return (node.get("title") if node else None) or movie_node_id

# ========= UI =========
st.set_page_config(page_title="RacoGraph – Recomendador por Grafos", layout="wide")
st.title("🎬 RacoGraph — Recomendação por Grafos (MovieLens Small)")

with st.spinner("Carregando grafo e dados..."):
    g, movies_df, ratings_df = load_data()
    id_to_title, title_to_id = build_title_maps(movies_df)

st.sidebar.header("⚙️ Parâmetros")
mode = st.sidebar.radio("Modo", ["Recomendar para usuário", "Filmes similares"])

metric = st.sidebar.selectbox(
    "Métrica",
    ["jaccard", "cosine"],
    index=0,
    help="Escolha Jaccard (mais simples) ou Cosine (usa as notas como peso)."
)

# ==== FILTRO DE GÊNERO ====
all_genres = sorted(
    {g for gs in movies_df["genres"] for g in str(gs).split("|") if g and g != "(no genres listed)"}
)
selected_genre = st.sidebar.selectbox(
    "🎭 Filtrar por gênero (opcional)",
    ["Todos"] + all_genres,
    index=0
)

# Sliders com títulos curtos + tooltips
min_co = st.sidebar.slider(
    "🔗 Nível de conexão",
    1, 10, 3, 1,
    help="Quanto maior, mais usuários em comum os filmes precisam ter."
)
topk = st.sidebar.slider(
    "🎬 Quantidade de recomendações",
    5, 50, 10, 1,
    help="Número de filmes exibidos na lista final."
)
min_user_rating = st.sidebar.slider(
    "⭐ Nota mínima",
    0.0, 5.0, 3.5, 0.5,
    help="Filmes avaliados abaixo disso não entram no cálculo do usuário."
)
k_similar = st.sidebar.slider(
    "🎯 Filmes relacionados usados",
    5, 100, 30, 5,
    help="Quantos filmes similares considerar por cada filme que o usuário viu."
)

colL, colR = st.columns([2, 3])

with colL:
    st.subheader("📊 Resumo do Grafo")
    n_users = sum(1 for nid in g.nodes if str(nid).startswith("U"))
    n_movies = sum(1 for nid in g.nodes if str(nid).startswith("F"))
    n_genres = sum(1 for nid in g.nodes if str(nid).startswith("G"))
    n_edges = sum(len(v) for v in g.adj.values()) // 2
    st.metric("Usuários (U)", n_users)
    st.metric("Filmes (F)", n_movies)
    st.metric("Gêneros (G)", n_genres)
    st.metric("Arestas", n_edges)

with colR:
    st.subheader("📥 Seleção")

    if mode == "Filmes similares":
        title = st.selectbox("Escolha um filme", sorted(title_to_id.keys()))
        movie_id = title_to_id[title]
        mid_node = f"F{int(movie_id)}"
        run = st.button("🔎 Buscar similares")

        if run:
            sims = topk_similar_movies(g, mid_node, k=topk, metric=metric, min_co=min_co)
            if not sims:
                st.warning("Nenhum similar encontrado com os parâmetros. Tente reduzir o nível de conexão ou usar 'jaccard'.")
            else:
                df = pd.DataFrame(
                    [{"movieId": m, "title": get_title(g, movies_df, m), "similarity": s} for m, s in sims]
                )
                st.write(f"**Top {len(df)} similares a:** `{mid_node}` — *{title}*")
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("title")["similarity"])

    else:
        user_id = st.selectbox("Escolha um usuário", sorted(ratings_df["userId"].unique().tolist()))
        uid_node = f"U{int(user_id)}"
        run = st.button("✨ Recomendar")

        if run:
            recs = recommend_for_user(
                g, uid_node,
                k_similar=k_similar, topn=topk,
                metric=metric, min_co=min_co, min_user_rating=min_user_rating
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
                    [{"movieId": f"F{int(x)}", "title": get_title(g, movies_df, f"F{int(x)}"), "score": None}
                     for x in popular_ids]
                )
            else:
                df = pd.DataFrame(
                    [{"movieId": m, "title": get_title(g, movies_df, m), "score": s} for m, s in recs]
                )

            st.write(f"**Recomendações para:** `{uid_node}`"
                     + (f" — filtro: *{selected_genre}*" if selected_genre != "Todos" else ""))
            st.dataframe(df, use_container_width=True)
            if "score" in df and df["score"].notna().any():
                st.bar_chart(df.set_index("title")["score"])
