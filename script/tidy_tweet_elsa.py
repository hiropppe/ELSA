import click
import math
import numpy as np
import json

from collections import defaultdict, OrderedDict
from filter_utils import SPECIAL_TOKENS
from operator import itemgetter
from tqdm import tqdm

#configure
cur_lan = "jp" # "de" or "fr" or "jp"
#input_file_name = "tmoji_tokens_%s" % cur_lan # precessed tweets after running the scripts in process_raw
input_file_name = '../process_raw_tweet/elsa_ja_top_emoji'
file_name = ['../process_raw_tweet/elsa_ja_processed']
vocab_path = "/data/elsa/vocab/" # path to store the numpy version of processed training and testing tweets
pre_vocab_file = vocab_path + "jp_vocab.json"
top_num = 64
#end configure

emoji = defaultdict(lambda: 0, {})
with open(input_file_name, "r") as stream:
    for line in stream:
        word, num = line.split('\t')
        try:
            emoji[word] += int(num)
        except KeyError:
            emoji[word] = int(num)
    stream.close()

sorted_emoji = sorted(emoji.items(), key=lambda x: x[1], reverse=True)
total = [0, 0]
wanted_emoji = []
with open(vocab_path + "emoji_%s" % cur_lan , "w") as f:
    for i, emoji in enumerate(sorted_emoji):
        f.write("%s\t%d\n" % (emoji[0], emoji[1]))
        if i < top_num:
            total[1] += emoji[1]
            wanted_emoji.append(emoji[0])
        total[0] += emoji[1]
print(wanted_emoji)
open("emoji_%s_top%s.json" % (cur_lan, top_num), "w").write( json.dumps( wanted_emoji  )  )
with open(vocab_path + "want_tmoji_%s" % cur_lan, "w") as f:
    for i, emoji in enumerate(sorted_emoji):
        if i < top_num:
            f.write("%s\t%d\n" % (emoji[0], emoji[1]))
open("%semoji_%s_top%s.json" % (vocab_path, cur_lan, top_num), "w").write( json.dumps( wanted_emoji  )  )

wanted_emoji = json.loads( open("%semoji_%s_top%s.json" % (vocab_path, cur_lan, top_num),"r").read() )
emoji_filter = wanted_emoji
tidy_data = []
for f in file_name:
    with open(f, "r") as stream:
        for line in stream:
            #data = line.strip().split("\t")
            #data = line.strip()
            data = json.loads(line)
            #if data in emoji_filter:
            if any(emoji in data for emoji in emoji_filter):
                tidy_data.append(data)
                #if len(json.loads(data[1])) > 2:
                #    tidy_data.append((data[0], json.loads(data[1])))
print(wanted_emoji, emoji_filter, len(wanted_emoji), tidy_data[0])

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
    # Calculate max length of sequences considered
    # Adjust batch_size accordingly to prevent GPU overflow
    #lengths = [len(t) for _, t in texts]
    lengths = [len(t) for t in texts]
    maxlen = roundup(np.percentile(lengths, 80.0))
    batch_size = 250 if maxlen <= 100 else 50
    print("mean: ", np.mean(lengths), "median: ", np.median(lengths), len(lengths), "avg: ", np.average(lengths))
    print("batch_size: ", batch_size, "maxlen:", maxlen)
    return batch_size, maxlen
batch_size, maxlen = calculate_batchsize_maxlen(tidy_data)

vocab_index = json.loads(open(pre_vocab_file, "r").read())
n_sentences = len(tidy_data)
fixed_length = maxlen
def find_tokens(words):
    assert len(words) > 0
    tokens = []
    for w in words:
        try:
            tokens.append(vocab_index[w])
        except KeyError:
            tokens.append(1)
    return tokens
infos = []
tokens = np.zeros((n_sentences, fixed_length), dtype='uint32')
next_insert = 0
n_ignored_unknowns = 0
#for s_info, s_words in tidy_data:
for s_words in tidy_data:
    s_tokens = find_tokens(s_words)
    if len(s_tokens) > fixed_length:
        s_tokens = s_tokens[:fixed_length]
    tokens[next_insert,:len(s_tokens)] = s_tokens
    tmp_info = np.zeros(64)
    for w in s_words:
        try:
            e_i = wanted_emoji.index(w)
            #print(w, e_i)
            tmp_info[e_i] = 1
            break
        except:
            continue
         
    assert tmp_info.sum() == 1
    infos.append(tmp_info)
    next_insert += 1
del tidy_data

#balance the input
balance_emoji = {x:[] for x in range(64)}
print(len(infos), len(tokens))
for i, info in enumerate(infos):
    k = np.argmax(info)
    balance_emoji[k].append(i)
train, val, test = [], [], []
for item in balance_emoji:
    line = balance_emoji[item]
    np.random.shuffle(line)
    length = len(line)
    train += line[:int(length*0.7)]
    val += line[int(length*0.7):int(length*0.9)]
    test += line[int(length*0.9):]
np.random.shuffle(train), np.random.shuffle(test), np.random.shuffle(val)
filter_token = []
filter_info = []
print(train[:5], test[:5], val[:5])
total = train + test + val
for index in total:
    filter_token.append(tokens[index])
    filter_info.append(infos[index])
print(len(filter_info), len(filter_token))
# finally processed info and label as emoji tweets
np.save(vocab_path + "%s_labels" % cur_lan, filter_info)
np.save(vocab_path + "%s_input" % cur_lan, filter_token)



def most_common_emoji(emoji_freq_path, topn):
    freq = {line.split()[0]: line.split()[1] for line in open(emoji_freq_path).readlines()}
    freq_topn = sorted(freq.items(), key=itemgetter(1), reverse=True)[:topn]
    emoji_topn = {freq[0] for freq in freq_topn}
    return emoji_topn


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
@click.argument("emoji_freq_path")
@click.argument("token2index_path")
@click.option("--topn", "-n", default=64)
@click.option("--traiin_size", "-vs", default=0.7)
@click.option("--test_size", "-ts", default=0.1)
def main(input_path, emoji_freq_path, token2index_path, topn, train_size, test_size):
    emoji_topn = most_common_emoji(emoji_freq_path, topn=topn)

    # filter out of topn emoji sentences
    with open(input_path, "r") as fi:
        tidy_data = []
        for line in tqdm(fi):
            sent = json.loads(line)
            if any(emoji in sent for emoji in emoji_topn):
                tidy_data.append(sent)

    batch_size, maxlen = calculate_batchsize_maxlen(tidy_data)

    token2index = json.loads(open(token2index_path, "r").read())

    def as_ids(tokens):
        tokens_as_id = []
        for token in tokens:
            try:
                tokens_as_id.append(token2index[token])
            except KeyError:
                tokens_as_id.append(SPECIAL_TOKENS.index("CUSTOM_UNKNOWN"))
        return tokens_as_id

    X = np.zeros((len(tidy_data), maxlen), dtype='uint32')
    y = []
    emoji_indices = defaultdict(list)
    for i, tokens in enumerate(tidy_data):
        tokens_as_id = as_ids(tokens)
        if len(tokens_as_id) > maxlen:
            tokens_as_id = tokens_as_id[:maxlen]
        X[i, :len(tokens_as_id)] = tokens_as_id
        each_y = np.zeros(topn)
        for token in tokens:
            emoji_index = emoji_topn.index(token)
            each_y[emoji_index] = 1
            break
        emoji_indices[emoji_index].append(i)

        assert each_y.sum() == 1
        y.append(each_y)

    tidy_data.clear()

    train, val, test = [], [], []
    val_size = 1-train_size-test_size
    for emoji, sample_indices in emoji_indices.items():
        np.random.shuffle(sample_indices)
        sample_length = len(sample_indices)
        train += sample_indices[:int(sample_length*train_size)]
        val += sample_indices[int(sample_length*train_size):int(sample_length*(train_size+val_size))]
        test += sample_indices[int(sample_length*(1-train_size-val_size)):]

    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)

    filter_token = []
    filter_info = []
    print(train[:5], test[:5], val[:5])
    total = train + test + val
    for index in total:
        filter_token.append(tokens[index])
        filter_info.append(infos[index])
    print(len(filter_info), len(filter_token))
    # finally processed info and label as emoji tweets
    np.save(vocab_path + "%s_labels" % cur_lan, filter_info)
    np.save(vocab_path + "%s_input" % cur_lan, filter_token)


if __name__ == '__main__':
    main()
