import click
import json
import os
import re
import sys
import unicodedata

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


CRLF_RE = re.compile(r'[\s\u3000]+')
RETWEET_RE = re.compile(r'^[rR][tT]')
HASHING_RE = re.compile(r'#[^\s]+')
MENTION_RE = re.compile(r'@[a-zA-Z0-9_]+:?')
# URL_RE = re.compile(r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
URL_RE = re.compile(r'(?:url\s*)?(?:https?://|\w*\.\w+\.\w+)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)…,]|[\u4E00-\u9FD0]|[あ-ん]|[\u30A1-\u30F4]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
TWEETLINK_RE = re.compile(r't.co/[a-zA-Z0-9]+')
PIC_TWITTER_RE = re.compile(r'pic.twitter.com/.+')


def prepro(json_line, lang=None, retweet=False):
    try:
        data = json.loads(json_line.strip())
        if lang and data["lang"] != lang:
            return None

        text = data['text']
        if retweet:
            text = RETWEET_RE.sub('', text)
        elif RETWEET_RE.match(text):
            return None

        text = unicodedata.normalize('NFKC', text)
        text = CRLF_RE.sub(' ', text)
        # text = HASHING_RE.sub('', text)
        text = text.replace('#', '')
        text = MENTION_RE.sub('', text)
        text = URL_RE.sub('', text)
        text = TWEETLINK_RE.sub('', text)
        text = PIC_TWITTER_RE.sub('', text)
        text = text.strip()
        return text
    except Exception:
        return None


@click.command()
@click.option("--lang", "-lang", help="")
@click.option("--retweet", "-rt", is_flag=True, default=True, help="")
@click.option("--processes", "-p", default=os.cpu_count()-1, help="")
@click.option("--chunksize", "-c", default=100, help="")
def main(lang, retweet, processes, chunksize):
    if processes > 1:
        pool = Pool(processes)
        map_func = partial(pool.imap_unordered, chunksize=chunksize)
    else:
        pool = None
        map_func = map

    prepro_func = partial(prepro, lang=lang, retweet=retweet)

    for text in tqdm(map_func(prepro_func, sys.stdin)):
        if text:
            try:
                print(text, file=sys.stdout)
            except Exception:
                pass

if __name__ == '__main__':
    main()
