import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

from absl import flags
from pathlib import Path

flags.DEFINE_string("lang", default="ja", help="lang to train")

flags.DEFINE_integer("maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("epoch_size", default=25000, help="number of data to process in each epoch")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")
flags.DEFINE_integer("n_classes", default=64, help="number of emoji classes to train")

flags.DEFINE_integer("lstm_hidden", default=512, help="")
flags.DEFINE_float("lstm_drop", default=0.5, help="")
flags.DEFINE_float("final_drop", default=0.5, help="")
flags.DEFINE_float("embed_drop", default=0.0, help="")
flags.DEFINE_bool("highway", default=False, help="")

FLAGS = flags.FLAGS


class DataGenerator():

    def __init__(self, data_dir):
        self.wv_path = (Path(data_dir) / ("{:s}_wv.npy".format(FLAGS.lang))).__str__()
        self.X_path = (Path(data_dir) / ("{:s}_X.npy".format(FLAGS.lang))).__str__()
        self.y_path = (Path(data_dir) / ("{:s}_y.npy".format(FLAGS.lang))).__str__()


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":
    tf.app.run()
