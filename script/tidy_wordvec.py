import click
import gensim
import json
import logging
import multiprocessing
import subprocess

from gensim.models import word2vec
from pathlib import Path
from tqdm import tqdm


class JsonSentenceGenerator():

    def __init__(self, input_path, n_splits):
        self.input_path = input_path
        self.rows = int(subprocess.check_output(['wc', '-l', input_path]).split()[0])
        self.n_splits = max(n_splits, 1)
        self.batch_size = self.rows // self.n_splits

    def generate(self):
        batch = 0
        sents = []
        with open(self.input_path, "r") as fi:
            n_sents = 0
            for i, line in tqdm(enumerate(fi)):
                sents.append(json.loads(line))
                n_sents += 1
                if not (n_sents < self.batch_size and batch == self.n_splits - 1):
                    yield sents
                    batch += 1
                    sents.clear()
                    n_sents = 0
            if sents:
                yield sents


@click.command()
@click.argument("data_dir")
@click.argument("lang")
@click.option('--json_input/--no_json_input', default=False)
@click.option("--input_model", "-m", help="")
@click.option("--n_splits", "-s", default=1, help="")
@click.option('--sg', default=1, help='Training algorithm: 1 for skip-gram; otherwise CBOW.')
@click.option('--size', default=200, help='Dimensionality of the word vectors.')
@click.option('--hs', default=0, help='If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.')
@click.option('--negative', default=20, help='If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.')
@click.option('--workers', default=None, help='Use these many worker threads to train the model (=faster training with multicore machines).')
def main(data_dir,
         lang,
         json_input,
         input_model,
         n_splits,
         sg,
         size,
         hs,
         negative,
         workers):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    workers = workers if workers else multiprocessing.cpu_count() - 1

    data_dir = Path(data_dir)
    input_path = (data_dir / "{:s}_tokens.txt".format(lang)).__str__()
    model_out = (data_dir / "{:s}_wv.model".format(lang)).__str__()
    wv_out = (data_dir / "{:s}_wv.txt".format(lang)).__str__()

    if input_model:
        model = gensim.models.Word2Vec.load(input_model)
        model.model_trimmed_post_training = False
        update = True
    else:
        model = gensim.models.Word2Vec(sg=sg,
                                       size=size,
                                       hs=hs,
                                       negative=negative,
                                       workers=workers)
        update = False

    if json_input:
        jsg = JsonSentenceGenerator(input_path, n_splits)
        for sents in jsg.generate():
            model.build_vocab(sents, update=update)
            model.train(sents, total_examples=model.corpus_count, epochs=model.iter)
            update = True
    else:
        corpus = word2vec.LineSentence(input_path)
        model = word2vec.Word2Vec(corpus,
                                  sg=sg,
                                  size=size,
                                  hs=hs,
                                  negative=negative,
                                  workers=workers)

    model.init_sims(replace=True)
    model.save(model_out)
    model.wv.save_word2vec_format(wv_out, binary=False)


if __name__ == '__main__':
    main()
