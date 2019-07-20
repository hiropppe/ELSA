from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from keras.optimizers import Adam
from operator import itemgetter
from pathlib import Path
from model import elsa_doc_model
from absl import flags
import keras
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()


flags.DEFINE_string("train_data", default=None, help="")
flags.DEFINE_string("valid_data", default=None, help="")
flags.DEFINE_string("test_data", default=None, help="")
flags.DEFINE_string("s_lang", default="en", help="lang")
flags.DEFINE_string("t_lang", default="ja", help="lang")
flags.DEFINE_integer("s_maxlen", default=20, help="")
flags.DEFINE_integer("t_maxlen", default=50, help="")

flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("batch_size", default=32, help="batch size")
flags.DEFINE_float("validation_split", default=0.2, help="")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("checkpoint_prefix", default="elsa", help="")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")

flags.DEFINE_integer("hidden_dim", default=64, help="")
flags.DEFINE_float("drop", default=0.5, help="")

FLAGS = flags.FLAGS


class TFRecordGenerator():

    def __init__(self, data_path, batch_size=32, s_maxlen=20, t_maxlen=20):
        self.dataset = tf.data.TFRecordDataset([data_path])
        self.batch_size = batch_size
        self.s_maxlen = s_maxlen
        self.t_maxlen = t_maxlen
        self.sess = tf.keras.backend.get_session()

    def get_n_features(self):
        iterator = self.dataset.map(self.parse_example).make_one_shot_iterator()
        get_next = iterator.get_next()
        data = self.sess.run(get_next)
        s_depth = data["src"].shape[1]
        t_depth = data["tgt"].shape[1]
        return [s_depth, t_depth]

    def get_n_steps(self):
        iterator = self.dataset.map(self.parse_example).make_one_shot_iterator()
        get_next = iterator.get_next()
        i = 0
        try:
            while True:
                self.sess.run(get_next)
                i += 1
        except tf.errors.OutOfRangeError:
            pass
        return i

    def parse_example(self, example):
        features = tf.io.parse_single_example(
            example,
            features={
                'src': tf.io.VarLenFeature(tf.float32),
                # 'src': tf.FixedLenFeature([], tf.float32, default_value=0.0),
                'tgt': tf.io.VarLenFeature(tf.float32),
                # 'tgt': tf.FixedLenFeature([], tf.float32, default_value=0.0),
                'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            })

        src = tf.reshape(features["src"].values, (self.s_maxlen, -1))
        tgt = tf.reshape(features["tgt"].values, (self.t_maxlen, -1))
        label = features["label"]
        return {"src": src, "tgt": tgt, "label": label}

    def generate(self):
        iterator = (self.dataset
                    .map(self.parse_example)
                    .shuffle(1000)
                    .batch(self.batch_size)
                    .repeat(1)
                    .make_one_shot_iterator())
        get_next = iterator.get_next()
        try:
            while True:
                data = tf.keras.backend.get_session().run(get_next)
                yield [data["src"], data["tgt"]], data["label"]
        except tf.errors.OutOfRangeError:
            pass

    def generate_inputs(self):
        iterator = (self.dataset
                    .map(self.parse_example)
                    .batch(self.batch_size)
                    .repeat(1)
                    .make_one_shot_iterator())
        get_next = iterator.get_next()
        try:
            while True:
                data = self.sess.run(get_next)
                yield [data["src"], data["tgt"]]
        except tf.errors.OutOfRangeError:
            pass

    def get_outputs(self):
        iterator = (self.dataset
                    .map(self.parse_example)
                    .batch(self.batch_size)
                    .repeat(1)
                    .make_one_shot_iterator())
        get_next = iterator.get_next()
        labels = []
        try:
            while True:
                data = self.sess.run(get_next)
                labels.append(data["label"])
        except tf.errors.OutOfRangeError:
            pass
        return np.array(labels)


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.train_data:
        train_data_generator = TFRecordGenerator(
            FLAGS.train_data, FLAGS.batch_size, FLAGS.s_maxlen, FLAGS.t_maxlen)
        train_data = train_data_generator.generate()
        n_features = train_data_generator.get_n_features()
        steps_per_epoch = train_data_generator.get_n_steps()
        if FLAGS.valid_data:
            valid_data_generator = TFRecordGenerator(
                FLAGS.valid_data, FLAGS.batch_size, FLAGS.s_maxlen, FLAGS.t_maxlen)
            validation_data = valid_data_generator.generate()
            validation_steps = valid_data_generator.get_n_steps()
        else:
            validation_data = None
            validation_steps = None

    if FLAGS.test_data:
        test_data_generator = TFRecordGenerator(
            FLAGS.test_data, FLAGS.batch_size, FLAGS.s_maxlen, FLAGS.t_maxlen)
        test_inputs = test_data_generator.generate_inputs()
        test_outputs = test_data_generator.get_outputs()
        n_features = test_data_generator.get_n_features()
        test_steps = test_data_generator.get_n_steps()

    model = elsa_doc_model(hidden_dim=FLAGS.hidden_dim,
                           dropout=FLAGS.drop,
                           nb_maxlen=[FLAGS.t_maxlen, FLAGS.s_maxlen],
                           nb_feature=n_features,
                           test=FLAGS.train_data is None)
    model.summary()

    checkpoint_dir = Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    checkpoint_weight_path = checkpoint_dir / \
        "{:s}_{:s}_{:s}.hdf5".format(FLAGS.checkpoint_prefix, FLAGS.s_lang, FLAGS.t_lang)

    if FLAGS.train_data:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                          patience=5, verbose=0, mode='auto'),
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_weight_path.__str__(), verbose=0, save_best_only=True, monitor='val_acc')
        ]
        model.compile(loss='binary_crossentropy', optimizer=FLAGS.optimizer, metrics=['accuracy'])
        model.fit_generator(train_data,
                            steps_per_epoch=steps_per_epoch,
                            epochs=FLAGS.epochs,
                            validation_data=validation_data,
                            validation_steps=validation_steps,
                            verbose=1,
                            callbacks=callbacks)

    if FLAGS.test_data:
        if not FLAGS.train_data:
            model.load_weights(filepath=checkpoint_weight_path)
        predict_total = model.predict_generator(test_inputs, steps=test_steps)
        predict_total = [int(x > 0.5) for x in predict_total]
        acc = accuracy_score(predict_total, test_outputs)
        print("Test Accuracy: {:.3f}\n".format(acc))
        print(classification_report(test_outputs, predict_total))


if __name__ == "__main__":
    tf.app.run()
