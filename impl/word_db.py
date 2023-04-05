# SPDX-License-Identifier: MIT
import sqlite3

import numpy as np

from . import resource

class WordDB:
    '''
    Word embeddings database.
    '''
    def __init__(self, path=None):
        if path is None:
            path = resource.filename('word-embeddings.db')
        self.con = sqlite3.connect(path)

        cur = self.con.cursor()
        try:
            cur.execute("CREATE TABLE embeddings(id INTEGER PRIMARY KEY, word TEXT NOT NULL, embedding BLOB NOT NULL, UNIQUE(word))")
        except sqlite3.OperationalError:
            pass # already exists
        else:
            self.con.commit()

    def insert(self, word, embedding):
        key = word.lower()
        data = embedding.astype(np.float16).tobytes()
        cur = self.con.cursor()
        cur.execute("INSERT INTO embeddings (word, embedding) VALUES (?, ?)", [key, data])
        self.con.commit()

    def lookup(self, word):
        key = word.lower()
        cur = self.con.cursor()
        cur.execute('SELECT embedding FROM embeddings WHERE word=?', [key])
        result = cur.fetchone()
        if result is not None:
            return np.frombuffer(result[0], dtype=np.float16).astype(np.float32)
        else:
            return None

