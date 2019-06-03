import click
import math
import numpy as np
import json

from collections import defaultdict, OrderedDict
from filter_utils import SPECIAL_TOKENS
from operator import itemgetter
from pathlib import Path
from tqdm import tqdm


def calculate_batchsize_maxlen(texts):
    """ Calculates the maximum length in the provided texts and a suitable
        batch size. Rounds up maxlen to the nearest multiple of ten.
    # Arguments:
        texts: List of inputs.
    # Returns:
        Batch size,
        max length
    """
    def roundup(x):
        return int(math.ceil(x / 10.0)) * 10

    print("calculate batch_size and maxlen")
    # Calculate max length of sequences considered
    # Adjust batch_size accordingly to prevent GPU overflow
    lengths = [len(t) for t in texts]
    maxlen = roundup(np.percentile(lengths, 80.0))
    batch_size = 250 if maxlen <= 100 else 50
    print("mean: ", np.mean(lengths), "median: ", np.median(lengths), len(lengths), "avg: ", np.average(lengths))
    print("batch_size: ", batch_size, "maxlen:", maxlen)
    return batch_size, maxlen


@click.command()
@click.argument("input_path")
@click.argument("output_dir")
@click.argument("prefix")
@click.argument("emoji_freq_path")
@click.argument("vocab_path")
@click.option("--topn", "-n", default=64)
@click.option("--train_size", "-vs", default=0.7)
@click.option("--test_size", "-ts", default=0.1)
def main(input_path, output_dir, prefix, emoji_freq_path, vocab_path, topn, train_size, test_size):

    out_X_path = Path(output_dir).joinpath("{:s}_X.npy".format(prefix)).as_posix()
    out_y_path = Path(output_dir).joinpath("{:s}_y.npy".format(prefix)).as_posix()

    token2index = json.loads(open(vocab_path, "r").read())

    def most_common_emoji(emoji_freq_path, topn):
        freq = {line.split()[0]: int(line.split()[1]) for line in open(emoji_freq_path).readlines()}
        freq_topn = sorted(freq.items(), key=itemgetter(1), reverse=True)[:topn]
        emoji_topn = [token2index[freq[0]] for freq in freq_topn]
        return emoji_topn

    def as_ids(tokens):
        tokens_as_id = []
        for token in tokens:
            try:
                tokens_as_id.append(token2index[token])
            except KeyError:
                tokens_as_id.append(SPECIAL_TOKENS.index("CUSTOM_UNKNOWN"))
        return tokens_as_id

    emoji_topn = most_common_emoji(emoji_freq_path, topn=topn)

    # filter out of topn emoji sentences
    with open(input_path, "r") as fi:
        tidy_data = []
        for line in tqdm(fi):
            tokens = line.split()
            id_tokens = as_ids(tokens)
            if any(emoji in id_tokens for emoji in emoji_topn):
                tidy_data.append(id_tokens)

    batch_size, maxlen = calculate_batchsize_maxlen(tidy_data)

    X = np.zeros((len(tidy_data), maxlen), dtype='uint32')
    y = []
    emoji_indices = defaultdict(list)
    for i, id_tokens in enumerate(tidy_data):
        each_y = np.zeros(topn)
        for token_id in id_tokens:
            try:
                emoji_index = emoji_topn.index(token_id)
                each_y[emoji_index] = 1
                break
            except ValueError:
                continue

        assert each_y.sum() == 1
        y.append(each_y)

        id_tokens = [t for t in id_tokens if t not in emoji_topn]
        X[i, :len(id_tokens)] = id_tokens[:min(maxlen, len(id_tokens))]

        emoji_indices[emoji_index].append(i)

    tidy_data.clear()

    train, val, test = [], [], []
    val_size = 1-train_size-test_size
    for emoji, sample_indices in emoji_indices.items():
        np.random.shuffle(sample_indices)
        sample_length = len(sample_indices)
        train += sample_indices[:int(sample_length*train_size)]
        val += sample_indices[int(sample_length*train_size):int(sample_length*(train_size+val_size))]
        test += sample_indices[int(sample_length*(train_size+val_size)):]
        print(sample_length,
              len(sample_indices[:int(sample_length*train_size)]), 
              len(sample_indices[int(sample_length*train_size):int(sample_length*(train_size+val_size))]),
              len(sample_indices[int(sample_length*(train_size+val_size)):]))

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    filtered_X = []
    filtered_y = []
    print(train[:5], test[:5], val[:5])
    total = train + val + test
    for index in total:
        filtered_X.append(X[index])
        filtered_y.append(y[index])
    print(len(filtered_y), len(filtered_X))
    # finally processed info and label as emoji tweets
    np.save(out_X_path, filtered_X)
    np.save(out_y_path, filtered_y)


if __name__ == '__main__':
    main()
