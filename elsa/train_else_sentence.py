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

flags.DEFINE_integer("batch_size", default=250, help="batch size")
flags.DEFINE_string("optimizer", default="adam", help="optimizer")
flags.DEFINE_float("lr", default=3e-4, help="learning rate")
flags.DEFINE_string("loss", default="categorical_crossentropy", help="loss")
flags.DEFINE_integer("epochs", default=100, help="max epochs")
flags.DEFINE_integer("patience", default=3, help="number of patience epochs for early stopping")
flags.DEFINE_string("checkpoint_dir", default="./ckpt", help="")
flags.DEFINE_string("data_dir", default="/data/elsa", help="directory contains preprocessed data")

flags.DEFINE_integer("lstm_hidden", default=512, help="")
flags.DEFINE_float("lstm_drop", default=0.5, help="")
flags.DEFINE_float("final_drop", default=0.5, help="")
flags.DEFINE_float("embed_drop", default=0.0, help="")
flags.DEFINE_bool("highway", default=False, help="")
flags.DEFINE_bool("multilabel", default=False, help="")

flags.mark_flags_as_required(["lang"])

FLAGS = flags.FLAGS


def main(unused_argv):
    del unused_argv

    tf.logging.set_verbosity(tf.logging.INFO)

    data_dir = Path(FLAGS.data_dir)
    wv_path = (data_dir / "{:s}_wv.npy".format(FLAGS.lang)).__str__()
    X_path = (data_dir / "{:s}_X.npy".format(FLAGS.lang)).__str__()
    y_path = (data_dir / "{:s}_y.npy".format(FLAGS.lang)).__str__()
    emoji_path = (data_dir / "{:s}_emoji.txt".format(FLAGS.lang)).__str__()

    wv = np.load(wv_path, allow_pickle=True)
    input_vec = np.load(X_path, allow_pickle=True)
    input_label = np.load(y_path, allow_pickle=True)

    nb_tokens = len(wv)
    embed_dim = wv.shape[1]
    input_len = len(input_label)
    nb_classes = input_label.shape[1]
    maxlen = input_vec.shape[1]

    train_end = int(input_len*0.7)
    val_end = int(input_len*0.9)

    (X_train, y_train) = (input_vec[:train_end], input_label[:train_end])
    (X_val, y_val) = (input_vec[train_end:val_end], input_label[train_end:val_end])
    (X_test, y_test) = (input_vec[val_end:], input_label[val_end:])

    if FLAGS.multilabel:
        def to_multilabel(y):
            outputs = []
            for i in nb_classes:
                outputs.append(y[:, i])
            return outputs

        y_train = to_multilabel(y_train)
        y_val = to_multilabel(y_val)
        y_test = to_multilabel(y_test)

    model = elsa_architecture(nb_classes=nb_classes,
                              nb_tokens=nb_tokens,
                              maxlen=maxlen,
                              final_dropout_rate=FLAGS.final_drop,
                              embed_dropout_rate=FLAGS.embed_drop,
                              load_embedding=True,
                              pre_embedding=wv,
                              high=FLAGS.highway,
                              embed_dim=embed_dim,
                              multilabel=FLAGS.multilabel)

    model.summary()

    if FLAGS.multilabel:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"

    if FLAGS.optimizer == 'adam':
        adam = Adam(clipnorm=1, lr=FLAGS.lr)
        model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    elif FLAGS.optimizer == 'rmsprop':
        model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])

    checkpoint_dir = Path(FLAGS.checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
    checkpoint_weight_path = (checkpoint_dir / "elsa_{:s}.hdf5".format(FLAGS.lang)).__str__()

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
              verbose=1)

    freq = {line.split()[0]: int(line.split()[1]) for line in open(emoji_path).readlines()}
    freq_topn = sorted(freq.items(), key=itemgetter(1), reverse=True)[:nb_classes]

    if FLAGS.multilabel:
        from functools import reduce

        def concat_flatten(x, y):
            return np.concatenate([x.flatten(), y.flatten()])

        y_pred = model.predict([X_test], batch_size=FLAGS.batch_size)
        y_pred = [np.squeeze(p) for p in y_pred]

        y_test_1d = np.array(y_test).flatten()
        y_pred_1d = np.array(y_pred).flatten()
        print(f1_score(y_test_1d, y_pred_1d > 0.5))
        print(classification_report(y_test_1d, y_pred_1d > 0.5))

        gold, pred = [], []
        for i in range(len(X_test)):
            each_gold, each_pred = [], []
            for c in range(nb_classes):
                if y_test[c][i] == 1.0:
                    each_gold.append(c+1)
                else:
                    each_gold.append(0)
                if y_pred[c][i] > 0.5:
                    each_pred.append(c+1)
                else:
                    each_pred.append(0)
            gold.extend(each_gold)
            pred.extend(each_pred)

        target_name = [""] + [e[0] for e in freq_topn]
        print(classification_report(gold, pred, target_names=target_name))
    else:
        _, acc = model.evaluate(X_test, y_test, batch_size=FLAGS.batch_size, verbose=0)
        print(acc)

        y_pred = model.predict(X_test)
        print(classification_report(y_test.argmax(axis=1), y_pred.argmax(
            axis=1), target_names=[e[0] for e in freq_topn]))


if __name__ == "__main__":
    tf.app.run()
