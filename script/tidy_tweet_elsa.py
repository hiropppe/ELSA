import click
import math
import numpy as np
import json
import sys

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


def assign_data_index_in_balance(train, val, test, indices_by_emoji, emoji_indices, topn):
    sample_holds = [train, val, test]
    n = 0
    pbar = tqdm()
    while emoji_indices:
        emoji_index = topn - n % topn - 1
        sample_indices = indices_by_emoji[emoji_index]
        for hold in sample_holds:
            if sample_indices:
                sample_index = sample_indices.pop()
                if sample_index in emoji_indices:
                    hold.append(sample_index)
                    emoji_indices.remove(sample_index)
        n += 1
        pbar.update()


@click.command()
@click.argument("data_dir")
@click.argument("lang")
@click.option("--topn", "-n", default=64)
@click.option("--label", type=click.Choice(["multi", "first", "most_common", "diverse_sampling"]), default="first")
@click.option("--train_size", "-vs", default=0.7)
@click.option("--test_size", "-ts", default=0.1)
def main(data_dir, lang, topn, label, train_size, test_size):

    data_dir = Path(data_dir)
    input_path = (data_dir / "{:s}_tokens.txt".format(lang)).__str__()
    emoji_path = (data_dir / "{:s}_emoji.txt".format(lang)).__str__()
    vocab_path = (data_dir / "{:s}_vocab.json".format(lang)).__str__()
    out_X_path = (data_dir / "{:s}_X.npy".format(lang)).__str__()
    out_y_path = (data_dir / "{:s}_y.npy".format(lang)).__str__()

    token2index = json.loads(open(vocab_path, "r").read())
    # index2token = [item[0] for item in sorted(token2index.items(), key=itemgetter(1))]

    def get_emoji_topn(emoji_path, topn):
        freq = {line.split()[0]: int(line.split()[1]) for line in open(emoji_path).readlines()}
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

    emoji_topn = get_emoji_topn(emoji_path, topn=topn)

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
    # single emoji data index
    indices_by_emoji1 = defaultdict(list)
    for i, id_tokens in enumerate(tidy_data):
        each_y = np.zeros(topn)
        emoji_index_set = set()
        for token_id in id_tokens:
            try:
                emoji_index = emoji_topn.index(token_id)
                emoji_index_set.add(emoji_index)
                if label == "first":
                    break
            except ValueError:
                continue

        id_tokens = [t for t in id_tokens if t not in emoji_topn]
        X[i, :len(id_tokens)] = id_tokens[:min(maxlen, len(id_tokens))]

        if label == "diverse_sampling":
            # pick min doc freq emoji ever
            min_index, min_count = -1, sys.maxsize
            for emoji_index in emoji_index_set:
                n_indices = len(indices_by_emoji1[emoji_index])
                if n_indices < min_count:
                    min_index = emoji_index
                    min_count = n_indices
            indices_by_emoji1[min_index].append(i)
            each_y[min_index] = 1
        elif label == "most_common":
            most_common_emoji_index = emoji_topn[min(emoji_topn.index[emoji_index] for emoji_index in emoji_index_set)]
            each_y[most_common_emoji_index] = 1
            indices_by_emoji1[most_common_emoji_index].append(i)
        else:  # multi, all
            for emoji_index in emoji_index_set:
                each_y[emoji_index] = 1
                indices_by_emoji1[emoji_index].append(i)

        y.append(each_y)

    for i in range(topn):
        print(i, len(indices_by_emoji1[i]))
        #print(i, index2token[emoji_topn[i]], len(indices_by_emoji1[i]), len(indices_by_emoji2[i]))

    tidy_data.clear()

    train, val, test = [], [], []
    val_size = 1 - train_size - test_size
    for i, (emoji, sample_indices) in enumerate(indices_by_emoji1.items()):
        np.random.shuffle(sample_indices)
        sample_length = len(sample_indices)
        train += sample_indices[:int(sample_length*train_size)]
        val += sample_indices[int(sample_length*train_size):int(sample_length*(train_size+val_size))]
        test += sample_indices[int(sample_length*(train_size+val_size)):]

    # assing multiple emoji (topn co-occured) indices in balance
    #assign_data_index_in_balance(train, val, test, indices_by_emoji2, emoji2_indices, topn)

    np.random.shuffle(train)
    np.random.shuffle(val)
    np.random.shuffle(test)

    filtered_X = []
    filtered_y = []
    total = train + val + test
    for index in total:
        filtered_X.append(X[index])
        filtered_y.append(y[index])
    print(len(filtered_y), len(filtered_X))

    X = np.array(filtered_X, dtype=np.uint32)
    y = np.array(filtered_y, dtype=np.uint32)

    np.save(out_X_path, filtered_X)
    np.save(out_y_path, filtered_y)


if __name__ == '__main__':
    main()
