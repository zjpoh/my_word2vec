import pathlib

# Directories
DATA_DIR = pathlib.Path.home() / "my_word2vec_data"
DATA_DIR_RAW = DATA_DIR / "raw"

# Create directory if not exists
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)

# Data files
IMDB_RAW = DATA_DIR_RAW / "aclImdb_v1.tar.gz"
IMDB = DATA_DIR_RAW / "aclImdb"
IMDB_TRN_UNSUP = IMDB / "train/unsup"
