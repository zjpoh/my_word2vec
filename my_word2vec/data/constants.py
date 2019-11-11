from my_word2vec.constants import ROOT_DIR

# Directories
DATA_DIR = "~/my_word2vec_data"
DATA_DIR_RAW = DATA_DIR / "raw"

# Create directory if not exists
DATA_DIR_RAW.mkdir(parents=True, exist_ok=True)

# Data files
IMDB_RAW = DATA_DIR_RAW / "aclImdb_v1.tar.gz"
IMDB = IMDB_RAW / "aclImdb"
IMDB_TRN_UNSUP = IMDB_RAW / "aclImdb/train/unsup"
IMDB_TST_UNSUP = IMDB_RAW / "aclImdb/test/unsup"
