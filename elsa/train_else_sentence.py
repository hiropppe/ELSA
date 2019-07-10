import keras
import numpy as np
import tensorflow as tf

from absl import flags
from model import elsa_architecture
from pathlib import Path

from operator import itemgetter
from keras.optimizers import Adam
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score


flags.DEFINE_string("lang", default=None, help="lang to train")

flags.DEFINE_integer("maxlen", default=20, help="max sequence length")
flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_string("loss", default="categorical_crossentropy", help="loss")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("epoch_size", default=25000, help="number of data to process in each epoch")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")

flags.DEFINE_integer("lstm_hidden", default=512, help="")
flags.DEFINE_float("lstm_drop", default=0.5, help="")
flags.DEFINE_float("final_drop", default=0.5, help="")
flags.DEFINE_float("embed_drop", default=0.0, help="")
flags.DEFINE_bool("highway", default=False, help="")

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)
    wv_path = (data_dir / "{:s}_wv.npy".format(FLAGS.lang)).__str__()
    X_path = (data_dir / "{:s}_X.npy".format(FLAGS.lang)).__str__()
    y_path = (data_dir / "{:s}_y.npy".format(FLAGS.lang)).__str__()
    emoji_freq_path = (data_dir / "{:s}_emoji.txt")

    wv = np.load(wv_path, allow_pickle=True)
    input_vec, input_label = np.load(X_path, allow_pickle=True), np.load(y_path, allow_pickle=True)

    nb_tokens = len(wv)
    embed_dim = wv.shape[1]
    input_len = len(input_label)
    nb_classes = input_label.shape[1]

    train_end = int(input_len*0.7)
    val_end = int(input_len*0.9)

    (X_train, y_train) = (input_vec[:train_end], input_label[:train_end])
    (X_val, y_val) = (input_vec[train_end:val_end], input_label[train_end:val_end])
    (X_test, y_test) = (input_vec[val_end:], input_label[val_end:])

    model = elsa_architecture(nb_classes=nb_classes,
                              nb_tokens=nb_tokens,
                              maxlen=FLAGS.maxlen,
                              final_dropout_rate=FLAGS.final_drop,
                              embed_dropout_rate=FLAGS.embed_drop,
                              load_embedding=True,
                              pre_embedding=wv,
                              high=FLAGS.highway,
                              embed_dim=embed_dim)

    model.summary()

    if FLAGS.optimizer == 'adam':
        adam = Adam(clipnorm=1, lr=FLAGS.lr)
        model.compile(loss=FLAGS.loss, optimizer=adam, metrics=['accuracy'])
    elif FLAGS.optimizer == 'rmsprop':
        model.compile(loss=FLAGS.loss, optimizer='rmsprop', metrics=['accuracy'])

    checkpoint_dir = Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    checkpoint_weight_path = (checkpoint_dir / "elsa_sentence_{:s}.hdf5".format(FLAGS.lang)).__str__()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=FLAGS.patience, verbose=0, mode='auto'),
        keras.callbacks.ModelCheckpoint(checkpoint_weight_path, monitor='val_loss',
                                        verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]
    model.fit(X_train,
              y_train,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=True)

    _, acc = model.evaluate(X_test, y_test, batch_size=FLAGS.batch_size, verbose=0)
    print(acc)

    freq = {line.split()[0]: int(line.split()[1]) for line in open(emoji_freq_path).readlines()}
    freq_topn = sorted(freq.items(), key=itemgetter(1), reverse=True)[:nb_classes]

    y_pred = model.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(
        axis=1), target_names=[e[0] for e in freq_topn]))


if __name__ == "__main__":
    tf.app.run()
