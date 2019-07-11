import numpy as np
import json
import pandas as pd
import tensorflow as tf

from absl import flags
from keras.models import Model
from model import elsa_architecture
from pathlib import Path


flags.DEFINE_string("corpus_prefix", default=None, help="corpus to encode")
flags.DEFINE_string("delimiter", default=",", help="delimiter of corpus")
flags.DEFINE_string("weight_prefix", default=None, help="elsa model weight path")

flags.DEFINE_integer("s_maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("t_maxlen", default=50, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")
flags.DEFINE_string("output_dir", default="./embed", help="directory contains preprocessed data")

flags.DEFINE_integer("nb_classes", default=64, help="")

flags.mark_flags_as_required(["corpus_prefix", "weight_prefix"])

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)

    output_dir = Path(FLAGS.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    df = pd.read_csv(FLAGS.corpus_prefix + ".csv", delimiter=FLAGS.delimiter)
    s_lang, t_lang = df.columns[1], df.columns[2]

    for i, lang in enumerate((s_lang, t_lang)):
        wv_path = (data_dir / "{:s}_wv.npy".format(lang)).__str__()
        wv = np.load(wv_path)
        nb_tokens = len(wv)
        embed_dim = wv.shape[1]

        if i == 0:
            maxlen = FLAGS.s_maxlen
        else:
            maxlen = FLAGS.t_maxlen

        vocab_path = data_dir / "{:s}_vocab.json".format(lang)
        vocab = json.load(open(vocab_path.__str__()))

        output_X_path = output_dir / (Path(FLAGS.corpus_prefix).name + "_" + lang + "_X.npy").__str__()
        output_y_path = output_dir / (Path(FLAGS.corpus_prefix).name + "_y.npy").__str__()

        model = elsa_architecture(nb_classes=FLAGS.nb_classes,
                                  nb_tokens=nb_tokens,
                                  maxlen=maxlen,
                                  embed_dim=embed_dim,
                                  feature_output=False,
                                  return_attention=False,
                                  test=True)
        model.load_weights(FLAGS.weight_prefix + "_" + lang + ".hdf5", by_name=True)

        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('attlayer').output)
        intermediate_layer_model.summary()

        def to_sequences(docs, vocab, maxlen):
            D = []
            for doc_text in docs:
                doc = json.loads(doc_text)
                doc_sequence = []
                for sent in doc:
                    sent_sequence = []
                    for token in sent:
                        if token in vocab:
                            sent_sequence.append(vocab[token])
                        else:
                            sent_sequence.append(1)  # UNKNOWN
                    doc_sequence.append(sent_sequence)
                doc_sequence = tf.keras.preprocessing.sequence.pad_sequences(doc_sequence, maxlen=maxlen, padding="post", dtype="int32")
                D.append(doc_sequence)
            return D

        D = to_sequences(df[lang].values, vocab, maxlen)

        encoded_D = []
        for inp in D:
            out = intermediate_layer_model.predict(inp, batch_size=FLAGS.batch_size)
            encoded_D.append(out)
        np.save(output_X_path, encoded_D)

    # save label
    np.save(output_y_path, df["label"].values)


if __name__ == "__main__":
    tf.app.run()
