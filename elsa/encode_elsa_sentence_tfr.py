import numpy as np
import json
import pandas as pd
import tensorflow as tf

from absl import flags
from keras.models import Model
from model import elsa_architecture
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences


flags.DEFINE_string("data", default=None, help="data to encode")
flags.DEFINE_string("delimiter", default=",", help="delimiter of corpus")
flags.DEFINE_string("s_lang", default="en", help="lang")
flags.DEFINE_string("t_lang", default="ja", help="lang")
flags.DEFINE_string("s_weight", default=None, help="elsa model weight path")
flags.DEFINE_string("t_weight", default=None, help="elsa model weight path")
flags.DEFINE_integer("s_maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("t_maxlen", default=50, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")

flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")

flags.DEFINE_integer("nb_classes", default=64, help="")

flags.mark_flags_as_required(["data", "s_weight", "t_weight"])

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)

    data = Path(FLAGS.data)
    df = pd.read_csv(data.__str__(), delimiter=FLAGS.delimiter)
    s_lang, t_lang = df.columns[1], df.columns[2]

    langs = (s_lang, t_lang)
    weight = {s_lang: FLAGS.s_weight, t_lang: FLAGS.t_weight}
    maxlen = {s_lang: FLAGS.s_maxlen, t_lang: FLAGS.t_maxlen}
    wv = {}
    nb_tokens = {}
    embed_dim = {}
    vocab = {}
    model = {}
    for lang in langs:
        wv_path = (data_dir / "{:s}_wv.npy".format(langs[lang])).__str__()
        wv[lang] = np.load(wv_path)
        nb_tokens[lang] = len(wv)
        embed_dim[lang] = wv.shape[1]

        vocab_path = data_dir / "{:s}_vocab.json".format(langs[lang])
        vocab[langs[lang]] = json.load(open(vocab_path.__str__()))

        model = elsa_architecture(nb_classes=FLAGS.nb_classes,
                                  nb_tokens=nb_tokens[lang],
                                  maxlen=maxlen[lang],
                                  embed_dim=embed_dim[lang],
                                  feature_output=False,
                                  return_attention=False,
                                  test=True)
        model.load_weights(weight, by_name=True)

        intermediate_layer_model = Model(
            inputs=model.input, outputs=model.get_layer('attlayer').output)
        intermediate_layer_model.summary()
        model[lang] = intermediate_layer_model

    def get_elsa_input(json_doc, vocab, maxlen):
        doc = json.loads(json_doc)
        doc_sequence = []
        for sent in doc:
            sent_sequence = []
            for token in sent:
                if token in vocab:
                    sent_sequence.append(vocab[token])
                else:
                    sent_sequence.append(1)  # UNKNOWN
            doc_sequence.append(sent_sequence)
        doc_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            doc_sequence, maxlen=maxlen, padding="post", dtype="int32")
        return doc_sequence

    def serialize_example(elsa_src, elsa_tgt, label):
        elsa_src = pad_sequences([elsa_src], dtype=elsa_src.dtype, maxlen=maxlen[s_lang])
        elsa_tgt = pad_sequences([elsa_tgt], dtype=elsa_tgt.dtype, maxlen=maxlen(t_lang))
        feature = {
            "src":   tf.train.Feature(float_list=tf.train.FloatList(value=elsa_src.flatten())),
            "tgt":   tf.train.Feature(float_list=tf.train.FloatList(value=elsa_tgt.flatten())),
            "label":  tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    record_path = data_dir / (data.name[:data.name.rindex(".")] + ".tfrecord").__str__()
    with tf.python_io.TFRecordWriter(record_path) as writer:
        for row in tqdm(df.iterrows()):
            cols = row[1]
            doc = [cols[s_lang], cols[t_lang]]
            elsa_out = {}
            for lang in langs:
                elsa_inp = get_elsa_input(doc[lang], vocab[lang], maxlen[lang])
                elsa_out[lang] = intermediate_layer_model.predict(elsa_inp)
            example = serialize_example(elsa_out[s_lang], elsa_out[t_lang], cols.label)
            writer.write(example)


if __name__ == "__main__":
    tf.app.run()
