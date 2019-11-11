"""Download datasets
"""

import logging
import shutil
import urllib.request

from my_word2vec.data import path


def download_imdb():
    """Download IMDB dataset."""
    logger = logging.getLogger(__name__)

    if not path.IMDB_RAW.exists():
        logger.info("Downloading IMDB dataset.")
        urllib.request.urlretrieve(
            "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            path.IMDB_RAW
        )

    if not path.IMDB.exists():
        logger.info("Extracting IMDB dataset.")
        shutil.unpack_archive(path.IMDB_RAW, path.DATA_DIR_RAW)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    download_imdb()
