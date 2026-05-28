"""Build FEVER wiki.db (and optional kNN cache) for InFact.

This script focuses on FEVER Wikipedia dump only and does not require
fever train/test claim files.

Examples:
  python -m scripts.fever.build_wiki_db
  python -m scripts.fever.build_wiki_db --download true --force true --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
import sqlite3
from urllib.request import urlretrieve

from config.globals import data_base_dir
from infact.tools.search.wiki_dump import WikiDumpAPI
import infact.common.embedding as embedding_module
from sentence_transformers import SentenceTransformer as RealSentenceTransformer


FEVER_WIKI_URL = "https://fever.ai/download/fever/wiki-pages.zip"


class SentenceTransformerCompat:
    def __init__(self, model_name, *args, config_kwargs=None, **kwargs):
        # Current sentence-transformers releases used in this environment do not
        # accept config_kwargs, so drop it while preserving the rest of the API.
        self._model = RealSentenceTransformer(model_name, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._model, item)


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_wiki_pages(zip_path: str, wiki_pages_dir: str, download: bool) -> None:
    if os.path.isdir(wiki_pages_dir) and any(os.scandir(wiki_pages_dir)):
        return

    if not os.path.exists(zip_path):
        if not download:
            raise FileNotFoundError(
                "wiki-pages.zip not found and --download is false. "
                "Please place FEVER wiki-pages.zip under data/FEVER/ or set --download true."
            )
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        print(f"Downloading FEVER wiki pages: {FEVER_WIKI_URL}")
        urlretrieve(FEVER_WIKI_URL, zip_path)

    print(f"Extracting wiki pages to: {wiki_pages_dir}")
    os.makedirs(wiki_pages_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(wiki_pages_dir)


def remove_existing_outputs(db_path: str, title_knn: str, body_knn: str) -> None:
    for p in [db_path, title_knn, body_knn]:
        if os.path.exists(p):
            os.remove(p)


def has_articles_table(db_path: str) -> bool:
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='articles';")
        return cur.fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=str, default="true",
                        help="Download wiki-pages.zip from fever.ai when missing (true/false)")
    parser.add_argument("--force", type=str, default="false",
                        help="Overwrite existing wiki.db/title_knn/body_knn (true/false)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Multiprocessing workers for parsing wiki pages")
    args = parser.parse_args()

    download = parse_bool(args.download)
    force = parse_bool(args.force)

    fever_dir = os.path.join(data_base_dir, "FEVER")
    zip_path = os.path.join(fever_dir, "wiki-pages.zip")
    wiki_pages_dir = os.path.join(fever_dir, "wiki-pages")

    db_path = os.path.join(fever_dir, "wiki.db")
    title_knn = os.path.join(fever_dir, "title_knn.pckl")
    body_knn = os.path.join(fever_dir, "body_knn.pckl")

    ensure_wiki_pages(zip_path=zip_path, wiki_pages_dir=wiki_pages_dir, download=download)

    if os.path.exists(db_path) and not has_articles_table(db_path):
        print(f"Detected stale partial DB without articles table: {db_path}")
        force = True

    if force:
        print("--force enabled: removing existing DB/kNN files.")
        remove_existing_outputs(db_path, title_knn, body_knn)
    elif os.path.exists(db_path):
        print(f"wiki.db already exists: {db_path}")
        print("Use --force true if you want to rebuild.")
        return

    api = WikiDumpAPI()
    # WikiDumpAPI creates an empty sqlite file during initialization.
    # Remove that placeholder so _build_db() can create the real database.
    try:
        api.db.close()
    except Exception:
        pass
    if os.path.exists(db_path):
        os.remove(db_path)

    # Patch the embedding backend to ignore unsupported config_kwargs.
    embedding_module.SentenceTransformer = SentenceTransformerCompat

    # _build_db() uses self.embedding_model directly, so initialize it here.
    api._setup_embedding_model()
    print("Building FEVER wiki.db (this may take a long time)...")
    api._build_db(from_path=wiki_pages_dir, num_workers=args.num_workers)

    # Re-open and build kNN cache once DB is ready.
    api = WikiDumpAPI()
    print("FEVER wiki.db ready.")
    print(f"DB: {db_path}")
    print(f"kNN title: {title_knn}")
    print(f"kNN body: {body_knn}")


if __name__ == "__main__":
    main()
