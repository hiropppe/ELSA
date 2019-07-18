import keras
import numpy as np
import tensorflow as tf

from absl import flags
from model import elsa_doc_model
from pathlib import Path

from operator import itemgetter
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score


flags.DEFINE_string("data", default="./embed/books_train_review",
                    help="directory contains preprocessed data")
flags.DEFINE_string("s_lang", default="en", help="lang")
flags.DEFINE_string("t_lang", default="ja", help="lang")
flags.DEFINE_integer("s_maxlen", default=20, help="")
flags.DEFINE_integer("t_maxlen", default=50, help="")

flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("batch_size", default=32, help="batch size")
flags.DEFINE_float("validation_split", default=0.1, help="")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")

flags.DEFINE_bool("pad", default=False, help="padding inputs")
flags.DEFINE_integer("hidden_dim", default=64, help="")
flags.DEFINE_float("drop", default=0.5, help="")

flags.DEFINE_float("val_size", default=0.2, help="")
flags.DEFINE_integer("random_state", default=123, help="")

flags.DEFINE_bool("test", default=False, help="")

FLAGS = flags.FLAGS


class H5Dataset():

    def __init__(self, h5_path, s_lang, t_lang, batch_size, val_size=0.2, random_state=None):
        import h5py as h5
        self.data = h5.File(h5_path)
        self.s_lang = s_lang
        self.t_lang = t_lang
        if not random_state:
            random_state = np.random.randint(1234)
        np.random.seed(random_state)

        data_len = len(self.data["label"])
        indices = np.random.permutation(data_len)
        self.batch_size = batch_size

        self.train_indices = indices[int(data_len*self.val_size):]
        self.validation_indices = indices[:int(data_len*self.val_size)]
        self.steps_per_epoch = len(self.train_indices) // self.batch_size 
        self.validation_steps = len(self.validation_indices) // self.batch_size 

    def generate_train_data(self):
        return self.generate(self.train_indices, self.steps_per_epoch)

    def generate_validation_data(self):
        return self.generate(self.validation_indices, self.validation_steps)

    def generate(self, indices, steps):
        src_data = self.data[self.s_lang]
        tgt_data = self.data[self.t_lang]
        label_data = self.data["label"]
        next_i = 0
        for s in range(steps):
            next_indices = indices[next_i:next_i+self.batch_size]
            input1 = src_data[next_indices]
            input2 = tgt_data[next_indices]
            label = label_data[next_indices]
            yield [input1, input2], label
            next_i += 1


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    h5dataset = None
    if FLAGS.data.endswith(".hdf5"):
        h5dataset = H5Dataset(FLAGS.data, FLAGS.s_lang, FLAGS.t_lang, FLAGS.batch_size, FLAGS.val_size, FLAGS.random_state)
    else:
        source_embed_path = FLAGS.data + "_" + FLAGS.s_lang + "_X.npy"
        target_embed_path = FLAGS.data + "_" + FLAGS.t_lang + "_X.npy"
        label_path = FLAGS.data + "_y.npy"

        source_X = np.load(source_embed_path, allow_pickle=True)
        target_X = np.load(target_embed_path, allow_pickle=True)
        y = np.load(label_path)

        if FLAGS.pad:
            source_X = tf.keras.preprocessing.sequence.pad_sequences(
                source_X, dtype=np.float32, maxlen=FLAGS.s_maxlen)
            target_X = tf.keras.preprocessing.sequence.pad_sequences(
                target_X, dtype=np.float32, maxlen=FLAGS.t_maxlen)

    model = elsa_doc_model(hidden_dim=FLAGS.hidden_dim,
                           dropout=FLAGS.drop,
                           nb_maxlen=[FLAGS.t_maxlen, FLAGS.s_maxlen],
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
                                          patience=5, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_weight_path, verbose=0, save_best_only=True, monitor='val_acc')
        ]
        model.compile(loss='binary_crossentropy', optimizer=FLAGS.optimizer, metrics=['accuracy'])
        if h5dataset:
            train_data = h5dataset.generate_train_data()
            validation_data = h5dataset.generate_validation_data()
            steps_per_epoch = h5dataset.steps_per_epoch
            validation_steps = h5dataset.validation_steps
            model.fit_generator(train_data,
                                steps_per_epoch=steps_per_epoch,
                                epochs=FLAGS.epochs,
                                validation_data=validation_data,
                                validation_steps=validation_steps,
                                verbose=1,
                                callbacks=callbacks)
        else:
            model.fit([source_X, target_X], y,
                      batch_size=FLAGS.batch_size,
                      epochs=FLAGS.epochs,
                      validation_split=FLAGS.validation_split,
                      verbose=1,
                      callbacks=callbacks)
    else:
        model.load_weights(filepath=checkpoint_weight_path)
        predict_total = model.predict([source_X, target_X], batch_size=FLAGS.batch_size)
        predict_total = [int(x > 0.5) for x in predict_total]
        acc = accuracy_score(predict_total, y)
        print("Test Accuracy: {:.3f}\n".format(acc))
        print(classification_report(y, predict_total))


if __name__ == "__main__":
    tf.app.run()
