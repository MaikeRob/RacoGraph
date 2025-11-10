# RacoGraph

Sistema de recomendação de filmes baseado em grafos usando Random Walk com Reinício (Personalized PageRank).

## Requisitos

- Python 3.12+
- uv (recomendado) ou pip

## Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/MaikeRob/RacoGraph.git
cd RacoGraph

# Instale as dependências
uv sync
```

<details>
<summary>Instalação alternativa com pip</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou: .venv\Scripts\activate  # Windows
pip install -e .
```
</details>

## Como Usar

### Interface Web

```bash
uv run streamlit run app.py
```

Acesse `http://localhost:8501` no navegador para:
- Encontrar filmes similares
- Gerar recomendações personalizadas
- Visualizar grafos interativos

### Avaliação do Sistema

```bash
# Avaliação básica
uv run python eval.py

# Avaliação customizada
uv run python eval.py --k 10 --num-walks 3000 --walk-length 25 --split random
```

**Parâmetros disponíveis:**
- `--k`: Número de recomendações (padrão: 10)
- `--num-walks`: Quantidade de caminhadas (padrão: 1000)
- `--walk-length`: Tamanho de cada caminhada (padrão: 10)
- `--min-user-rating`: Nota mínima para preferência (padrão: 3.0)
- `--split`: Modo de divisão treino/teste (`last` ou `random`)

## Estrutura

```
RacoGraph/
├── app.py              # Interface Streamlit
├── eval.py             # Avaliação e métricas
├── recommender.py      # Algoritmo Random Walk
├── graph.py            # Estrutura de grafo
├── data_loader.py      # Carregamento de dados
├── visualizer.py       # Visualização interativa
├── constants.py        # Configurações
└── data/
    └── ml-latest-small/  # Dataset MovieLens
```

## Algoritmo

Utiliza **Random Walk com Reinício** para navegar pelo grafo de:
- Usuários → Filmes (avaliações)
- Filmes → Gêneros (classificações)

As recomendações são geradas através de caminhadas aleatórias ponderadas pelas avaliações dos usuários.

## Dataset

**MovieLens Small** (~100k avaliações):
- 610 usuários
- 9.742 filmes
- Avaliações, tags e metadados