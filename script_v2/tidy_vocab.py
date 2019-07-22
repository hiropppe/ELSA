import click
import gensim
import json
import numpy as np

from pathlib import Path
from filter_utils import SPECIAL_TOKENS


@click.command()
@click.argument("data_dir")
@click.argument("lang")
def main(data_dir, lang):
    data_dir = Path(data_dir)
    input_path = (data_dir / "{:s}_wv.txt".format(lang)).__str__()
    out_vocab_path = (data_dir / "{:s}_vocab.json".format(lang)).__str__()
    out_vec_path = (data_dir / "{:s}_wv.npy".format(lang)).__str__()

    vocab = SPECIAL_TOKENS.copy()

    with open(input_path, "r") as fi:
        dim = int(next(fi).split()[1])
        for i, line in enumerate(fi):
            data = line.strip().split(' ')
            if data[0] not in vocab:
                vocab.append(data[0])

    wv = gensim.models.KeyedVectors.load_word2vec_format(input_path, binary=False)
    tidy_wv = []
    for word in vocab:
        try:
            tidy_wv.append(wv[word])
        except KeyError:
            tidy_wv.append(np.random.uniform(-1, 1, dim))

    tidy_wv = np.vstack(tidy_wv)

    word2index = {w: i for i, w in enumerate(vocab)}
    open(out_vocab_path, "w").write(json.dumps(word2index))

    np.save(out_vec_path, tidy_wv)


if __name__ == '__main__':
    main()
