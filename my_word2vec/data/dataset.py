"""Download datasets
"""

import logging
import re
import shutil
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from my_word2vec.data import path


def download_imdb():
    """Download IMDB dataset."""
    logger = logging.getLogger(__name__)

    if not path.IMDB_RAW.exists():
        logger.info("Downloading IMDB dataset.")
        urllib.request.urlretrieve(
            "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            path.IMDB_RAW,
        )

    if not path.IMDB.exists():
        logger.info("Extracting IMDB dataset.")
        shutil.unpack_archive(path.IMDB_RAW, path.DATA_DIR_RAW)


def preprocess_sentence(sentence):
    """Preprocess sentence by lowercasing, remove all non-alphanumeric
    characters and stripping whitespace.

    Parameters
    ----------
    sentence : string

    Return
    ------
    string
        Processed sentence
    """

    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9]", r" ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.strip()

    return sentence


def read_imdb(n_reviews=10):
    """Read IMDB corpus and clean sentence using `preprocess_sentence`.

    Parameters
    ----------
    n_reviews : int
        Number of IMDB reviews to read

    Returns
    -------
    1D np.array of type string
        IMDB corpus, where every row is a new review.
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading imdb data.")

    download_imdb()

    corpus = [
        np.loadtxt(path.as_posix(), dtype=str, delimiter="\n", comments=None)
        for path in list(path.IMDB_TRN_UNSUP.iterdir())[:n_reviews]
    ]
    corpus = np.array(corpus)

    corpus = np.vectorize(preprocess_sentence)(corpus)

    return corpus


def train_test_splits(corpus, random_state=0):
    """Split corpus to train / valid / test.

    Parameters
    ----------
    corpus : List of string
    random_state : int or np.random.RandomState

    Returns
    -------
    List of string, List of string, List of string
        Train / valid / test corpus.
    """
    logger = logging.getLogger(__name__)
    logger.info("Splitting corpus to train / valid / test.")

    trn_corpus, tst_corpus = train_test_split(
        corpus, test_size=0.2, random_state=random_state
    )
    trn_corpus, vld_corpus = train_test_split(
        trn_corpus, test_size=0.25, random_state=random_state
    )

    return trn_corpus, vld_corpus, tst_corpus


def tokenize_keras(corpus, n_tokens=1000):
    """Tokenize corpus with keras text tokenizer.

    Parameters
    ----------
    corpus : List of string
    n_tokens : int
        Vocabulary size

    Returns
    -------
    List of int, tf.keras.preprocessing.text.Tokenizer
        Tokenized corpus, tokenizer that tokenizes the corpus
    """
    logger = logging.getLogger(__name__)
    logger.info("Tokenize corpus.")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=n_tokens, lower=False, filters=""
    )
    tokenizer.fit_on_texts(corpus)

    corpus = tokenizer.texts_to_sequences(corpus)

    corpus = tf.keras.preprocessing.sequence.pad_sequences(
        corpus, padding="post"
    )

    return corpus, tokenizer


def create_skip_gram_samples(corpus, window_size=2):
    """Create skip-gram context and target pairs.

    Parameters
    ----------
    corpus : List of string
    tokenizer : tf.keras.preprocessing.text.Tokenizer
    window_size : int
        Skip-gram window size

    Returns
    -------
    List of int, list of int
        List of context and list of targets
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating skip-gram context and target pair.")

    contexts = []
    targets = []
    for doc in corpus:
        for i, word in enumerate(doc):
            min_range = max(0, i - window_size)
            max_range = min(len(doc), i + window_size + 1)
            for j in range(min_range, i):
                contexts.append([word])
                targets.append([doc[j]])
            for j in range(i + 1, max_range):
                contexts.append([word])
                targets.append([doc[j]])

    return contexts, targets


def create_dataset(contexts, targets, batch_size=64, seed=0):
    """Create TensorFlow dataset with contexts and targets

    Parameters
    ----------
    contexts : List of int
    targets : List of int
    batch_size : int

    Returns
    -------
    tf.data.Dataset
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating TensorFlow Dataset.")

    dataset = (
        tf.data.Dataset.from_tensor_slices((contexts, targets))
        .shuffle(len(contexts), seed=seed)
        .batch(batch_size)
        .repeat()
    )

    return dataset


def imdb_etl(n_reviews=100, batch_size=64):
    """Run IDDB data etl to read data, to split to train / valid / test,
    tokenize, create context/target pair, and create TensorFlow Dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info("Run IMDB etl.")

    corpus = read_imdb(n_reviews=n_reviews)

    trn_corpus, vld_corpus, tst_corpus = train_test_splits(corpus)

    trn_corpus_tkn, tokenizer = tokenize_keras(trn_corpus)
    vld_corpus_tkn = tokenizer.texts_to_sequences(vld_corpus)
    tst_corpus_tkn = tokenizer.texts_to_sequences(tst_corpus)

    trn_x, trn_y = create_skip_gram_samples(trn_corpus_tkn, tokenizer)
    vld_x, vld_y = create_skip_gram_samples(vld_corpus_tkn, tokenizer)
    tst_x, tst_y = create_skip_gram_samples(tst_corpus_tkn, tokenizer)

    trn = create_dataset(trn_x, trn_y, batch_size)
    vld = create_dataset(vld_x, vld_y, batch_size)
    tst = create_dataset(tst_x, tst_y, batch_size)

    return trn, vld, tst, tokenizer, len(trn_x) // batch_size


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    imdb_etl()
