# dataset.py

import os
import glob
import dill
from .utils import sgf

class DataSet:
    def __init__(self, dir_name, batch_size, steps):
        self.batch_size = batch_size
        self.steps = steps

        cache_file = f"{dir_name}.dill"

        # 1) Try loading from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    self.sgf_data = dill.load(f)
                print(f"Loaded SGF cache ({len(self.sgf_data)} games) from {cache_file}")
            except Exception as e:
                print(f"Warning: failed to load cache {cache_file}: {e}. Will re-parse SGF.")
                self._parse_and_cache(dir_name, cache_file)
        else:
            # 2) No cache → parse & cache
            self._parse_and_cache(dir_name, cache_file)

        # ... the rest of your init (e.g. splitting into train/test, etc.) …

    def _parse_and_cache(self, dir_name, cache_file):
        """Parse all SGF under dir_name, skipping broken files, then cache."""
        print("Parsing SGF data (one-time)…")
        sgf_games = []

        # look for .sgf files in the directory
        pattern = os.path.join(dir_name, "*.sgf")
        for path in glob.glob(pattern):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    data = f.read()
                games = sgf.parse_from_string(data)
                sgf_games.extend(games)
            except Exception as e:
                print(f"  ⚠️  Skipping {os.path.basename(path)}: parse error: {e}")

        print(f"Parsed {len(sgf_games)} games; caching to {cache_file} …")
        try:
            with open(cache_file, "wb") as f:
                dill.dump(sgf_games, f)
        except Exception as e:
            print(f"Warning: could not write cache file {cache_file}: {e}")

        self.sgf_data = sgf_games

    # ... keep your other methods (batching, iteration, etc.) unchanged ...
