import click
import json
import os
import re
import unicodedata

from functools import partial
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


CRLF_RE = re.compile(r'[\s\u3000]+')
RETWEET_RE = re.compile(r'^[rR][tT]')
HASHING_RE = re.compile(r'#[^\s]+')
MENTION_RE = re.compile(r'@[a-zA-Z0-9_]+:?')
#URL_RE = re.compile(r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
URL_RE = re.compile(
    r'(?:url\s*)?(?:https?://|\w*\.\w+\.\w+)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)…,]|[\u4E00-\u9FD0]|[あ-ん]|[\u30A1-\u30F4]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
TWEETLINK_RE = re.compile(r't.co/[a-zA-Z0-9]+')
PIC_TWITTER_RE = re.compile(r'pic.twitter.com/.+')


class TextGenerator():

    def __init__(self, file_path, lang, retweet, processes, chunksize):
        self.file_path = file_path
        self.lang = lang
        self.retweet = retweet
        self.processes = processes
        self.chunksize = chunksize

    def extract_text(self, json_line):
        try:
            data = json.loads(json_line.strip())
            if self.lang and data["lang"] not in self.lang:
                return None, None

            text = data['text']
            if self.retweet:
                text = RETWEET_RE.sub('', text)
            elif RETWEET_RE.match(text):
                return None, None

            text = unicodedata.normalize('NFKC', text)
            text = CRLF_RE.sub(' ', text)
            # text = HASHING_RE.sub('', text)
            text = text.replace('#', '')
            text = MENTION_RE.sub('', text)
            text = URL_RE.sub('', text)
            text = TWEETLINK_RE.sub('', text)
            text = PIC_TWITTER_RE.sub('', text)
            text = text.strip()
            return text, data["lang"]
        except Exception:
            return None, None

    def __iter__(self):
        if self.processes > 1:
            pool = Pool(self.processes)
            map_func = partial(pool.imap_unordered, chunksize=self.chunksize)
        else:
            pool = None
            map_func = map

        try:
            with open(self.file_path) as stream:
                for text, lang in tqdm(map_func(self.extract_text, stream)):
                    if text:
                        yield text, lang
        finally:
            if pool is not None:
                pool.close()


@click.command()
@click.argument("input_path")
@click.argument("output_dir")
@click.option("--lang", "-lang", multiple=True, help="")
@click.option("--retweet", "-rt", is_flag=True, default=False, help="")
@click.option("--processes", "-p", default=os.cpu_count()-1, help="")
@click.option("--chunksize", "-c", default=100, help="")
def main(input_path, output_dir, lang, retweet, processes, chunksize):
    text_generator = TextGenerator(input_path,
                                   lang,
                                   retweet,
                                   processes,
                                   chunksize)

    output_dir = Path(output_dir)
    outputs = {}
    for each_lang in lang:
        outputs[each_lang] = open((output_dir / ("tweet_" + each_lang + ".txt")).__str__(), "w")

    for each_text, each_lang in text_generator:
        try:
            print(each_text, file=outputs[each_lang])
        except Exception:
            pass

    for each_lang in lang:
        outputs[each_lang].close()


if __name__ == '__main__':
    main()
