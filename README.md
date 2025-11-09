# RacoGraph

Sistema de recomendação de filmes baseado em grafos usando dados do MovieLens.

## Pré-requisitos

- Python 3.12+
- uv (gerenciador de pacotes Python)

## Instalação

1. Instale o uv (se ainda não tiver):

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Ou instale via pip:
```bash
pip install uv
```

2. Clone o repositório:
```bash
git clone https://github.com/MaikeRob/RacoGraph.git
cd RacoGraph
```

3. Instale as dependências:

**Usando uv (recomendado):**
```bash
uv sync
```

**Usando pip (alternativa):**

*Linux/macOS:*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

*Windows:*
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Como Rodar

**Interface Web (Streamlit):**

```bash
# Usando uv (recomendado)
uv run streamlit run app.py

# Ou com pip (após ativar o ambiente virtual)
streamlit run app.py
```

**Avaliação do Sistema:**

```bash
# Usando uv (recomendado)
uv run python eval.py --k 10 --split last

# Ou com pip (após ativar o ambiente virtual)
python eval.py --k 10 --split last
```

## Estrutura do Projeto

### 📁 Arquivos Principais

- **[`app.py`](app.py)** - Interface web interativa (Streamlit)
- **[`eval.py`](eval.py)** - Sistema de avaliação e métricas
- **[`recommender.py`](recommender.py)** - Algoritmo Random Walk com Reinício
- **[`graph.py`](graph.py)** - Estrutura de dados do grafo
- **[`data_loader.py`](data_loader.py)** - Carregamento de dados do MovieLens
- **[`constants.py`](constants.py)** - Constantes e configurações
- **[`data/ml-latest-small/`](data/ml-latest-small/)** - Dataset do MovieLens
- **[`pyproject.toml`](pyproject.toml)** - Configuração do projeto e dependências

## Dataset

O projeto utiliza o dataset MovieLens Small, que contém:
- Avaliações de filmes
- Tags de filmes
- Informações de filmes
- Links para IMDb e TMDb