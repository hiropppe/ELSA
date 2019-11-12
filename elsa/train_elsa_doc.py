import keras
import numpy as np
import tensorflow as tf
import warnings

from absl import flags
from keras.utils import Sequence
from model import elsa_doc_model
from pathlib import Path

from operator import itemgetter
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score


def warn(*args, **kwargs):
    pass


warnings.warn = warn


flags.DEFINE_string("data", default=None,
                    help="directory contains preprocessed data")
flags.DEFINE_string("s_lang", default="en", help="lang")
flags.DEFINE_string("t_lang", default="ja", help="lang")
flags.DEFINE_integer("s_maxlen", default=20, help="")
flags.DEFINE_integer("t_maxlen", default=50, help="")

flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("batch_size", default=32, help="batch size")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_bool("shuffle", default=True, help="whether to shuffle the training data before each epoch")

flags.DEFINE_bool("pad", default=True, help="padding inputs")
flags.DEFINE_integer("hidden_dim", default=64, help="")
flags.DEFINE_float("drop", default=0.5, help="")

flags.DEFINE_float("val_size", default=0.1, help="")
flags.DEFINE_integer("random_state", default=123, help="")

flags.DEFINE_bool("test", default=False, help="")

flags.mark_flags_as_required(["data"])

FLAGS = flags.FLAGS


class H5Generator(Sequence):

    def __init__(self, data, s_lang, t_lang, batch_size, indices, shuffle=True):
        self.indices = indices
        self.s_data = data[s_lang]
        self.t_data = data[t_lang]
        self.label_data = data["label"]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_indices = sorted(batch_indices.tolist())
        input1 = self.s_data[batch_indices]
        input2 = self.t_data[batch_indices]
        label = self.label_data[batch_indices]
        return [input1, input2], label

    def __len__(self):
        return len(self.indices) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class H5Dataset():

    def __init__(self, h5_path, s_lang, t_lang, batch_size, val_size=0.1, random_state=None):
        import h5py as h5
        self.data = h5.File(h5_path, mode="r", driver="core")
        self.s_lang = s_lang
        self.t_lang = t_lang
        if not random_state:
            random_state = np.random.randint(1234)
        np.random.seed(random_state)
        data_len = len(self.data["label"])
        self.batch_size = batch_size
        self.indices = np.random.permutation(data_len)
        self.train_indices = self.indices[int(data_len*val_size):]
        self.validation_indices = self.indices[:int(data_len*val_size)]

    def generate_from_train_data(self, shuffle=True):
        return H5Generator(self.data, self.s_lang, self.t_lang, self.batch_size, self.train_indices, shuffle=shuffle)

    def generate_from_validation_data(self):
        return H5Generator(self.data, self.s_lang, self.t_lang, self.batch_size, self.validation_indices, shuffle=False)

    def get_all_data(self):
        s_input = self.data[self.s_lang]
        t_input = self.data[self.t_lang]
        label = self.data["label"]
        return [s_input, t_input], label


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    h5dataset = None
    if FLAGS.data.endswith(".hdf5"):
        h5dataset = H5Dataset(FLAGS.data, FLAGS.s_lang, FLAGS.t_lang,
                              FLAGS.batch_size, FLAGS.val_size, FLAGS.random_state)
    else:
        data = np.load(FLAGS.data, allow_pickle=True)

        source_X = data[FLAGS.s_lang]
        target_X = data[FLAGS.t_lang]

        if FLAGS.pad:
            source_X = tf.keras.preprocessing.sequence.pad_sequences(
                source_X, dtype=np.float32, maxlen=FLAGS.s_maxlen)
            target_X = tf.keras.preprocessing.sequence.pad_sequences(
                target_X, dtype=np.float32, maxlen=FLAGS.t_maxlen)

        X = [source_X, target_X]
        y = data["label"]

    model = elsa_doc_model(hidden_dim=FLAGS.hidden_dim,
                           dropout=FLAGS.drop,
                           nb_maxlen=[FLAGS.s_maxlen, FLAGS.t_maxlen],
                           test=FLAGS.test)
    model.summary()

    checkpoint_dir = Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    checkpoint_weight_path = (
        checkpoint_dir / "elsa_{:s}_{:s}.hdf5".format(FLAGS.s_lang, FLAGS.t_lang)).__str__()

    if not FLAGS.test:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                          patience=FLAGS.patience, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_weight_path, verbose=0, save_best_only=True, monitor='val_loss')
        ]
        model.compile(loss='binary_crossentropy', optimizer=FLAGS.optimizer, metrics=['accuracy'])
        if h5dataset:
            train_data = h5dataset.generate_from_train_data(shuffle=FLAGS.shuffle)
            validation_data = h5dataset.generate_from_validation_data()
            model.fit_generator(train_data,
                                epochs=FLAGS.epochs,
                                validation_data=validation_data,
                                verbose=1,
                                callbacks=callbacks)
        else:
            model.fit(X, y,
                      batch_size=FLAGS.batch_size,
                      epochs=FLAGS.epochs,
                      validation_split=FLAGS.val_size,
                      verbose=1,
                      callbacks=callbacks,
                      shuffle=FLAGS.shuffle)
    else:
        if h5dataset:
            X, y = h5dataset.get_all_data()
        model.load_weights(checkpoint_weight_path)
        predict_total = model.predict(X, batch_size=FLAGS.batch_size)
        predict_total = [int(x > 0.5) for x in predict_total]
        acc = accuracy_score(predict_total, y)
        print("Test Accuracy: {:.3f}\n".format(acc))
        print(classification_report(y, predict_total))


if __name__ == "__main__":
    tf.app.run()
