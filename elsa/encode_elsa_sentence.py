import numpy as np
import json
import pandas as pd
import tensorflow as tf

from absl import flags
from model import elsa_architecture
from pathlib import Path

from keras.models import Model, Sequential
from operator import itemgetter
from keras.optimizers import Adam
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score


flags.DEFINE_string("corpus_prefix", help="corpus to encode")
flags.DEFINE_string("lang", default="en", help="lang to train")
flags.DEFINE_string("weight_prefix", help="elsa model weight path")

flags.DEFINE_integer("maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("epoch_size", default=25000, help="number of data to process in each epoch")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")
flags.DEFINE_string("output_dir", default="./embed", help="directory contains preprocessed data")

flags.DEFINE_integer("lstm_hidden", default=512, help="")
flags.DEFINE_float("lstm_drop", default=0.5, help="")
flags.DEFINE_float("final_drop", default=0.5, help="")
flags.DEFINE_float("embed_drop", default=0.0, help="")
flags.DEFINE_bool("highway", default=False, help="")
flags.DEFINE_integer("nb_classes", default=64, help="")

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)
    output_dir = Path(FLAGS.output_dir)
    df = pd.read_csv(FLAGS.corpus)
    s_lang, t_lang = df.columns[1], df.columns[2]

    for lang in (s_lang, t_lang):
        vocab_path = data_dir / "{:s}_vocab.json".format(lang)
        vocab = json.load(open(vocab_path.__str__()))

        output_X_path = output_dir / Path(FLAGS.corpus_prefix).name + FLAGS.lang + "_X.npy"
        output_y_path = output_dir / Path(FLAGS.corpus_prefix).name + "_y.npy"

        model = elsa_architecture(nb_classes=FLAGS.nb_classes,
                                  nb_tokens=FLAGS.nb_tokens,
                                  maxlen=FLAGS.maxlen,
                                  embed_dim=FLAGS.embed_dim,
                                  feature_output=False,
                                  return_attention=False,
                                  test=True)
        model.load_weights(FLAGS.weight_prefix + "_" + lang + ".hdf5", by_name=True)

        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('attlayer').output)
        intermediate_layer_model.summary()

        def to_sequences(texts, vocab, maxlen):
            sequences = []
            for text in texts:
                tokens = text.split()
                sequence = []
                for token in enumerate(tokens):
                    if token in vocab:
                        sequence.append(vocab[token])
                    else:
                        sequence.append(1)  # UNKNOWN
            sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding="post", dtype="int32")
            return sequences

        sequences = to_sequences(df[lang].values, vocab, FLAGS.maxlen)

        encoded_sentences = intermediate_layer_model.predict(sequences, batch_size=FLAGS.batch_size)
        np.save(output_X_path, encoded_sentences)

    # save label
    np.save(output_y_path, df["label"].values)


if __name__ == "__main__":
    tf.app.run()
