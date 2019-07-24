import numpy as np
import json
import os
import pandas as pd
import tensorflow as tf

from absl import flags
from keras.models import Model
from model import elsa_architecture
from pathlib import Path
from tqdm import tqdm


flags.DEFINE_string("data", default=None, help="data to encode")
flags.DEFINE_string("delimiter", default=",", help="delimiter of corpus")
flags.DEFINE_string("s_weight", default=None, help="elsa model weight path")
flags.DEFINE_string("t_weight", default=None, help="elsa model weight path")
flags.DEFINE_integer("s_maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("t_maxlen", default=50, help="max sequence length")
flags.DEFINE_integer("nb_classes", default=64, help="")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")
flags.DEFINE_string("embed_dir", default="./embed", help="directory contains preprocessed data")
flags.DEFINE_bool("h5", default=False, help="save using hdf5") 

flags.mark_flags_as_required(["data", "s_weight", "t_weight"])

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)

    embed_dir = Path(FLAGS.embed_dir)
    if not embed_dir.exists():
        embed_dir.mkdir()

    data = Path(FLAGS.data)
    df = pd.read_csv(data.__str__(), delimiter=FLAGS.delimiter)
    s_lang, t_lang = df.columns[1], df.columns[2]

    weight = {s_lang: FLAGS.s_weight, t_lang: FLAGS.t_weight}
    maxlen = {s_lang: FLAGS.s_maxlen, t_lang: FLAGS.t_maxlen}

    output_prefix = data.name[:data.name.rindex(".")]

    if FLAGS.h5:
        import h5py as h5
        output_path = embed_dir / (output_prefix + ".hdf5")
        tmp_output_path = embed_dir / (".tmp." + output_prefix + ".hdf5")
        if tmp_output_path.exists():
            os.remove(tmp_output_path.__str__())
        h5f = h5.File(tmp_output_path.__str__())
    else:
        embed = {}

    for i, lang in enumerate((s_lang, t_lang)):
        wv_path = (data_dir / "{:s}_wv.npy".format(lang)).__str__()
        wv = np.load(wv_path)
        nb_tokens = len(wv)
        embed_dim = wv.shape[1]

        vocab_path = data_dir / "{:s}_vocab.json".format(lang)
        vocab = json.load(open(vocab_path.__str__()))

        model = elsa_architecture(nb_classes=FLAGS.nb_classes,
                                  nb_tokens=nb_tokens,
                                  maxlen=maxlen[lang],
                                  embed_dim=embed_dim,
                                  feature_output=False,
                                  return_attention=False,
                                  test=True)
        model.load_weights(weight[lang], by_name=True)

        intermediate_layer_model = Model(
            inputs=model.input, outputs=model.get_layer('attlayer').output)
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
                doc_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                    doc_sequence, maxlen=maxlen, padding="post", dtype="int32")
                D.append(doc_sequence)
            return D

        D = to_sequences(df[lang].values, vocab, maxlen[lang])

        if FLAGS.h5:
            for j, inp in tqdm(enumerate(D)):
                out = intermediate_layer_model.predict(inp, batch_size=FLAGS.batch_size)
                out = tf.keras.preprocessing.sequence.pad_sequences(
                    [out], maxlen=maxlen[lang], dtype=out.dtype)[0]
                if j == 0:
                    depth = out.shape[1]
                    dataset = h5f.require_dataset(
                            lang,
                            dtype=out.dtype,
                            shape=(1, maxlen[lang], depth),
                            maxshape=(None, maxlen[lang], depth),  # 'None' == arbitrary size
                            exact=False,
                            chunks=(32, maxlen[lang], depth),
                            compression="lzf")
                if j >= len(dataset):
                    dataset.resize((j+1, maxlen[lang], depth))
                dataset[j] = out
        else:
            #output_X_path = (embed_dir / (output_prefix + "_" + lang + "_X.npy")).__str__()
            encoded_D = []
            for inp in tqdm(D):
                out = intermediate_layer_model.predict(inp)
                encoded_D.append(out)
            embed[lang] = encoded_D
            #np.save(output_X_path, encoded_D)

    if FLAGS.h5:
        h5f.create_dataset("label", data=df["label"].values)
        h5f.close()
        os.rename(tmp_output_path.__str__(), output_path.__str__())
    else:
        #output_y_path = (embed_dir / (output_prefix + "_y.npy")).__str__()
        np.savez(output_path, s_lang=embed[s_lang], t_lang=embed[t_lang], label=df["label"].values)
        #np.save(output_y_path, df["label"].values)


if __name__ == "__main__":
    tf.app.run()
