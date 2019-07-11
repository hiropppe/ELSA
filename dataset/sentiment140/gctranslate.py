import sqlite3
import sys
import time
import traceback

from google.cloud import translate


class GCTranslate():

    ddl = """
        CREATE TABLE IF NOT EXISTS gc_translate_cache (
            input TEXT NOT NULL,
            tgt TEXT NOT NULL,
            output TEXT NOT NULL,
            src TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(input, tgt)
        )
    """

    def __init__(self, cache_path=None, local=False):
        if not cache_path and local:
            raise ValueError('local mode is enabled without cache_path')
        self.client = translate.Client()
        self.cache_path = cache_path
        self.local = local
        if cache_path:
            self.conn = sqlite3.connect(cache_path)
            self.conn.cursor().execute(self.ddl)
        else:
            self.conn = None
        self.n_requests = 0
        self.cache_hits = 0
        self.n_gct_requests = 0

    def __del__(self):
        if self.n_requests:
            print('Cache hit ratio: {:.4f}'.format(
                self.cache_hits/self.n_requests), file=sys.stdout)
        if self.conn:
            self.conn.close()

    def translate(self, inp, tgt, src=None, model='nmt'):
        self.n_requests += 1
        if self.conn:
            cursor = self.conn.cursor()
            translatedText = self.lookup_cache(cursor, inp, tgt)
            if translatedText:
                return translatedText

            if self.local:
                return None

            translatedText = self.google_cloud_translate(inp, tgt, src, model)

            try:
                if src is None:
                    src = ''
                if model is None:
                    model = 'nmt'
                cursor.execute("""
                    REPLACE INTO gc_translate_cache
                        (input, tgt, output, src, model, created_at)
                    VALUES
                        (?, ?, ?, ?, ?, ?)
                """, (inp, tgt, translatedText, src, model, time.time()))
                self.conn.commit()
            except Exception:
                traceback.print_exc()

            return translatedText
        else:
            return self.google_cloud_translate(inp, tgt, src, model)

    def google_cloud_translate(self, inp, tgt, src, model):
        self.n_gct_requests += 1
        translation = self.client.translate(
            inp, target_language=tgt, source_language=src, model=model)
        return translation['translatedText']

    def lookup_cache(self, cursor, inp, tgt):
        cursor.execute(
            'SELECT output FROM gc_translate_cache WHERE input = ? and tgt = ?', (inp, tgt))
        data = cursor.fetchone()
        if data:
            self.cache_hits += 1
            return data[0]
        else:
            return None

    def print_stats(self):
        print('requests: {:d} google requests {:d} cache hits: {:.3f}'.format(
            self.n_requests, self.n_gct_requests, self.cache_hits/self.n_requests if self.n_requests else 0))
