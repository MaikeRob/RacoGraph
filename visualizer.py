"""Visualização de recomendações RacoGraph usando grafos interativos."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from graph import Graph, is_user, is_movie, is_genre


def capture_random_walks(
    g: Graph,
    start_nodes: List[str],
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    weights: Dict[str, float] = None
) -> Tuple[Dict[str, int], List[List[str]]]:
    """Executa random walks capturando caminhos e contando visitas."""
    if not start_nodes:
        return {}, []

    if weights is None:
        weights = {node: 1.0 for node in start_nodes}

    total_weight = sum(weights.values())
    start_probs = {node: weights[node] / total_weight for node in start_nodes}

    visit_counts: Dict[str, int] = defaultdict(int)
    walk_paths: List[List[str]] = []

    # Armazena mais caminhos para melhor visualização (mas ainda limitado)
    max_paths_to_store = min(100, num_walks)
    store_every = max(1, num_walks // max_paths_to_store)

    for walk_idx in range(num_walks):
        current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
        path = [current]

        for _ in range(walk_length):
            visit_counts[current] += 1

            if random.random() < restart_prob:
                current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
                path.append(current)
                continue

            neighbors = g.get_neighbors(current)
            if not neighbors:
                current = random.choices(start_nodes, weights=[start_probs[n] for n in start_nodes])[0]
                path.append(current)
                continue

            next_nodes = []
            edge_weights = []

            for edge in neighbors:
                next_node = edge.get("node")
                edge_weight = float(edge.get("weight", 1.0))
                next_nodes.append(next_node)
                edge_weights.append(edge_weight)

            total_edge_weight = sum(edge_weights)
            if total_edge_weight > 0:
                edge_probs = [w / total_edge_weight for w in edge_weights]
                current = random.choices(next_nodes, weights=edge_probs)[0]
            else:
                current = random.choice(next_nodes)

            path.append(current)

        # Armazena caminho (amostragem)
        if walk_idx % store_every == 0:
            walk_paths.append(path)

    return dict(visit_counts), walk_paths


def create_recommendation_visualization(
    g: Graph,
    user_id: str,
    start_movies: List[str],
    movie_weights: Dict[str, float],
    recommendations: List[Tuple[str, float]],
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    movies_df,
    output_file: str = "recommendation_viz.html"
) -> str:
    """
    Cria visualização HTML das recomendações mostrando caminhos e nós mais visitados.

    Args:
        g: Grafo
        user_id: ID do usuário
        start_movies: Filmes iniciais (avaliados pelo usuário)
        movie_weights: Pesos dos filmes iniciais (ratings)
        recommendations: Lista de recomendações [(movie_id, score), ...]
        num_walks: Número de caminhadas executadas
        walk_length: Comprimento das caminhadas
        restart_prob: Probabilidade de reinício
        movies_df: DataFrame com informações dos filmes
        output_file: Nome do arquivo HTML de saída

    Returns:
        Caminho absoluto do arquivo gerado
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("ERRO: PyVis não instalado. Execute: pip install pyvis")
        return ""

    print("\nGerando visualização de recomendações...")

    # Captura caminhadas com caminhos
    visit_counts, walk_paths = capture_random_walks(
        g, start_movies, num_walks, walk_length, restart_prob, movie_weights
    )

    # Identifica nós mais visitados (top 30%)
    sorted_visits = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)
    top_30_percent = max(1, len(sorted_visits) * 30 // 100)
    highly_visited = {node for node, _ in sorted_visits[:top_30_percent]}

    # Identifica filmes recomendados
    recommended_movies = {movie_id for movie_id, _ in recommendations[:10]}

    # Coleta nós dos caminhos percorridos
    relevant_nodes = {user_id} | set(start_movies) | recommended_movies

    connected_nodes = set()
    for path in walk_paths:
        if any(node in start_movies for node in path) or user_id in path:
            connected_nodes.update(path)

    relevant_nodes.update(connected_nodes)

    # Limita tamanho se muito grande
    if len(relevant_nodes) > 200:
        priority_nodes = {user_id} | set(start_movies) | recommended_movies
        other_nodes = sorted(relevant_nodes - priority_nodes,
                            key=lambda n: visit_counts.get(n, 0), reverse=True)
        keep_count = max(20, len(other_nodes) * 30 // 100)
        relevant_nodes = priority_nodes | set(other_nodes[:keep_count])

    print(f"   Visualizando {len(relevant_nodes)} nós conectados ao usuário")    # Cria rede
    net = Network(height="900px", width="100%", bgcolor="#ffffff",
                  font_color="#000000", notebook=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    # Helper para obter título do filme
    def get_movie_title(movie_id: str) -> str:
        try:
            mid = int(movie_id[1:])
            title = movies_df.loc[movies_df["movieId"] == mid, "title"].iloc[0]
            return title[:40] + "..." if len(title) > 40 else title
        except (ValueError, IndexError, KeyError):
            return movie_id

    # Adiciona nós
    for node_id in relevant_nodes:
        attrs = g.nodes.get(node_id, {})
        visits = visit_counts.get(node_id, 0)

        # Define cor e tamanho
        if node_id == user_id:
            color, size, border_width = '#FF0000', 40, 5
            label = f"USER {node_id}"
        elif node_id in recommended_movies:
            color, size, border_width = '#00CC00', 25 + min(visits / 10, 20), 4
            label = get_movie_title(node_id)
        elif node_id in start_movies:
            color, size, border_width = '#4169E1', 20 + min(visits / 10, 15), 3
            label = get_movie_title(node_id)
        elif is_movie(node_id):
            if node_id in highly_visited:
                color, size, border_width = '#FF8C00', 18 + min(visits / 10, 12), 2
            else:
                color, size, border_width = '#87CEEB', 12 + min(visits / 10, 8), 1
            label = get_movie_title(node_id)
        elif is_genre(node_id):
            color, size, border_width = '#FFD700', 30, 2
            label = attrs.get('name', node_id)
        elif is_user(node_id):
            color, size, border_width = '#FFB6C1', 12, 1
            label = node_id
        else:
            color, size, border_width = '#CCCCCC', 10, 1
            label = node_id

        # Tooltip
        title_parts = [f"<b>{node_id}</b>", f"Tipo: {attrs.get('type', 'Unknown')}"]
        if visits > 0:
            title_parts.append(f"Visitas: {visits}")
        if node_id in recommended_movies:
            title_parts.append("<b>RECOMENDADO</b>")
        elif node_id in start_movies:
            title_parts.append(f"Avaliação: {movie_weights.get(node_id, 0):.1f}")
        if is_movie(node_id) and (full_title := get_movie_title(node_id)) != node_id:
            title_parts.append(f"Título: {full_title}")
        tooltip = "<br>".join(title_parts)

        net.add_node(
            node_id,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            borderWidth=border_width,
            borderWidthSelected=border_width + 2
        )

    # Coleta arestas dos caminhos
    path_edges = set()
    for path in walk_paths:
        if any(node in start_movies for node in path) or user_id in path:
            for i in range(len(path) - 1):
                if path[i] in relevant_nodes and path[i+1] in relevant_nodes:
                    path_edges.add(tuple(sorted([path[i], path[i+1]])))

    # Adiciona arestas
    added_edges = set()
    for u_id, v_id in path_edges:
        edge_key = tuple(sorted([u_id, v_id]))
        if edge_key in added_edges or u_id not in relevant_nodes or v_id not in relevant_nodes:
            continue

        weight = next((e.get('weight', 1.0) for e in g.adj.get(u_id, []) if e['node'] == v_id), 1.0)

        net.add_edge(u_id, v_id, value=float(weight),
                    title=f"Peso: {weight:.2f} | Caminho",
                    color='rgba(255, 0, 0, 0.6)', width=2)
        added_edges.add(edge_key)

    print(f"   {len(added_edges)} arestas dos caminhos")

    # Adiciona legenda como HTML customizado
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; background: white;
                border: 2px solid #333; padding: 15px; border-radius: 8px;
                font-family: Arial; font-size: 12px; z-index: 1000; max-width: 250px;">
        <h3 style="margin: 0 0 10px 0; font-size: 14px;">Legenda</h3>
        <div style="margin: 5px 0;"><span style="color: #FF0000;">●</span> <b>Usuário</b></div>
        <div style="margin: 5px 0;"><span style="color: #00CC00;">●</span> <b>Recomendações</b> (top 10)</div>
        <div style="margin: 5px 0;"><span style="color: #4169E1;">●</span> Filmes avaliados (início)</div>
        <div style="margin: 5px 0;"><span style="color: #FF8C00;">●</span> Filmes muito visitados</div>
        <div style="margin: 5px 0;"><span style="color: #87CEEB;">●</span> Outros filmes</div>
        <div style="margin: 5px 0;"><span style="color: #FFD700;">●</span> Gêneros</div>
        <div style="margin: 5px 0;"><span style="color: #FFB6C1;">●</span> Outros usuários</div>
        <hr style="margin: 10px 0;">
        <div style="margin: 5px 0;"><span style="color: red; font-weight: bold;">--</span> Caminhos percorridos</div>
        <div style="margin: 5px 0; font-size: 10px; color: #666;">
            Tamanho do nó = número de visitas<br>
            {num_walks} caminhadas × {walk_length} passos
        </div>
    </div>
    """
    legend_html = legend_html.replace("{num_walks}", str(num_walks)).replace("{walk_length}", str(walk_length))

    # Salva HTML
    output_path = Path(output_file)
    net.save_graph(str(output_path))

    # Adiciona legenda ao HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Insere legenda antes do </body>
    html_content = html_content.replace('</body>', f'{legend_html}</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    abs_path = output_path.absolute()
    print(f"   Visualização salva: {abs_path}")

    return str(abs_path)


# ==================== API Simplificada ====================

def visualize_recommendations(
    g: Graph,
    user_id: str,
    start_movies: List[str],
    movie_weights: Dict[str, float],
    recommendations: List[Tuple[str, float]],
    num_walks: int,
    walk_length: int,
    restart_prob: float,
    movies_df,
    output_file: str = "recommendation_viz.html"
) -> str:
    """
    API principal para gerar visualização de recomendações.

    Retorna o caminho do arquivo HTML gerado.
    """
    return create_recommendation_visualization(
        g, user_id, start_movies, movie_weights, recommendations,
        num_walks, walk_length, restart_prob, movies_df, output_file
    )
