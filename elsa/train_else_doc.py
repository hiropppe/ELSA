import keras
import numpy as np
import tensorflow as tf

from absl import flags
from model import elsa_doc_model
from pathlib import Path

from operator import itemgetter
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score


flags.DEFINE_string("s_lang", default="en", help="lang")
flags.DEFINE_string("t_lang", default="ja", help="lang")
flags.DEFINE_string("weight_path", help="elsa model weight path")

flags.DEFINE_string("mode", default="train")
flags.DEFINE_integer("maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("epoch_size", default=25000, help="number of data to process in each epoch")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")
flags.DEFINE_string("input_dir", default="./embed", help="directory contains preprocessed data")

flags.DEFINE_integer("hidden_dim", default=100, help="")
flags.DEFINE_float("drop", default=0.5, help="")

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    input_dir = Path(FLAGS.input_dir)

    source_embed_path = input_dir / (Path(FLAGS.input_prefix).name + FLAGS.s_lang + "_X.npy")
    target_embed_path = input_dir / (Path(FLAGS.input_prefix).name + FLAGS.t_lang + "_X.npy")
    label_path = input_dir / (Path(FLAGS.input_prefix).name + "_y.npy")

    source_X = np.load(source_embed_path, allow_pickle=True)
    target_X = np.load(target_embed_path, allow_pickle=True)
    y = np.load(label_path)

    model = elsa_doc_model(hidden_dim=FLAGS.hidden_dim, dropout=FLAGS.dropout, mode=FLAGS.mode)
    model.summary() 

    if FLAGS.mode == "train":
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(filepath=FLAGS.weight_path, verbose=0, save_best_only=True, monitor='val_acc')
        ]
        model.fit([source_X, target_X], y,
                  batch_size=FLAGS.batch_size,
                  epochs=FLAGS.epochs,
                  validation_split=FLAGS.validation_split,
                  verbose=0,
                  callbacks=callbacks)
    else:
        model.load_weights(filepath=FLAGS.weight_path)
        predict_total = model.predict([target_X, source_X], batch_size=FLAGS.batch_size)
        predict_total = [int(x > 0.5) for x in predict_total]
        acc = accuracy_score(predict_total, y)
        print("Test Accuracy: {:.3f}\n".format(acc))
        print(classification_report(y, predict_total))


if __name__ == "__main__":
    tf.app.run()
