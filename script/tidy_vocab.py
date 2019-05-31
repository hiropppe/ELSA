import click
import gensim
import json
import numpy as np

from pathlib import Path
from filter_utils import SPECIAL_TOKENS
from tqdm import tqdm


@click.command()
@click.argument("input_path")
@click.argument("output_dir")
@click.argument("prefix")
def main(input_path, output_dir, prefix):
    out_vocab_path = Path(output_dir).joinpath("{:s}_vocab.txt".format(prefix)).as_posix()
    out_vec_path = Path(output_dir).joinpath("{:s}_wv.npy".format(prefix)).as_posix()

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
