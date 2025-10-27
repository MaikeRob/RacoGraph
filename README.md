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
git clone <url-do-repositorio>
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

**Usando uv:**
```bash
uv run python main.py
```

**Usando pip (após ativar o ambiente virtual):**

*Linux/macOS:*
```bash
source .venv/bin/activate
python main.py
```

*Windows:*
```bash
.venv\Scripts\activate
python main.py
```

## Estrutura do Projeto

- [`main.py`](main.py) - Arquivo principal da aplicação
- [`graph.py`](graph.py) - Implementação do grafo de recomendações
- [`data/ml-latest-small/`](data/ml-latest-small/) - Dataset do MovieLens
- [`pyproject.toml`](pyproject.toml) - Configuração do projeto e dependências

## Dataset

O projeto utiliza o dataset MovieLens Small, que contém:
- Avaliações de filmes
- Tags de filmes
- Informações de filmes
- Links para IMDb e TMDb