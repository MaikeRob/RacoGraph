"""Constantes e configurações do projeto RacoGraph."""
from pathlib import Path

# Prefixos de Nós
NODE_PREFIX_USER = "U"
NODE_PREFIX_MOVIE = "F"
NODE_PREFIX_GENRE = "G"

# Tipos de Nós
NODE_TYPE_USER = "User"
NODE_TYPE_MOVIE = "Movie"
NODE_TYPE_GENRE = "Genre"

# Tipos de Relações
RELATION_RATED = "avaliou"
RELATION_BELONGS_TO_GENRE = "pertence_ao_genero"

# Caminhos de Dados
DATA_DIR = Path("data/ml-latest-small")
MOVIES_FILE = DATA_DIR / "movies.csv"
RATINGS_FILE = DATA_DIR / "ratings.csv"
TAGS_FILE = DATA_DIR / "tags.csv"
LINKS_FILE = DATA_DIR / "links.csv"

# Random Walk - Parâmetros Padrão
DEFAULT_RESTART_PROB_USER = 0.15
DEFAULT_RESTART_PROB_SIMILAR = 0.30
DEFAULT_NUM_WALKS = 1000
DEFAULT_WALK_LENGTH = 10

# Recomendação - Parâmetros Padrão
DEFAULT_MIN_USER_RATING = 4.0
DEFAULT_TOP_N = 10

# Valores Especiais
NO_GENRE_LABEL = "(no genres listed)"
