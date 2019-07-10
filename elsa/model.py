from attlayer import AttentionWeightedAverage
from copy import deepcopy
from operator import itemgetter
from keras.layers.merge import concatenate
from keras.layers import (
    Layer, Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D, LSTM, Activation,
    TimeDistributed, Highway
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import L1L2
from pathlib import Path
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from os.path import exists


def elsa_architecture(nb_classes,
                      nb_tokens,
                      maxlen,
                      feature_output=False,
                      embed_dropout_rate=0,
                      final_dropout_rate=0,
                      embed_dim=300,
                      embed_l2=1E-6,
                      return_attention=False,
                      load_embedding=False,
                      pre_embedding=None,
                      high=False,
                      LSTM_hidden=512,
                      LSTM_drop=0.5):
    """
    Returns the DeepMoji architecture uninitialized and
    without using the pretrained model weights.
    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        feature_output: If True the model returns the penultimate
                        feature vector rather than Softmax probabilities
                        (defaults to False).
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.
        high: use or not the highway network
    # Returns:
        Model with the given parameters.
    """
    class NonMasking(Layer):
        def __init__(self, **kwargs):
            self.supports_masking = True
            super(NonMasking, self).__init__(**kwargs)

        def build(self, input_shape):
            input_shape = input_shape

        def compute_mask(self, input, input_mask=None):
            # do not pass the mask to the next layers
            return None

        def call(self, x, mask=None):
            return x

        def get_output_shape_for(self, input_shape):
            return input_shape

    # define embedding layer that turns word tokens into vectors
    # an activation function is used to bound the values of the embedding
    model_input = Input(shape=(maxlen,), dtype='int32')
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None

    if not load_embedding and pre_embedding is None:
        embed = Embedding(input_dim=nb_tokens, output_dim=embed_dim, mask_zero=True, input_length=maxlen, embeddings_regularizer=embed_reg,
                          name='embedding')
    else:
        embed = Embedding(input_dim=nb_tokens, output_dim=embed_dim, mask_zero=True, input_length=maxlen, weights=[pre_embedding],
                          embeddings_regularizer=embed_reg, trainable=True, name='embedding')
    if high:
        x = NonMasking()(embed(model_input))
    else:
        x = embed(model_input)
    x = Activation('tanh')(x)

    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if embed_dropout_rate != 0:
        embed_drop = SpatialDropout1D(embed_dropout_rate, name='embed_drop')
        x = embed_drop(x)

    # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
    # ordering of the way the merge is done is important for consistency with the pretrained model
    lstm_0_output = Bidirectional(
        LSTM(LSTM_hidden, return_sequences=True, dropout=LSTM_drop), name="bi_lstm_0")(x)
    lstm_1_output = Bidirectional(
        LSTM(LSTM_hidden, return_sequences=True, dropout=LSTM_drop), name="bi_lstm_1")(lstm_0_output)
    x = concatenate([lstm_1_output, lstm_0_output, x])
    if high:
        x = TimeDistributed(Highway(activation='tanh', name="high"))(x)
    # if return_attention is True in AttentionWeightedAverage, an additional tensor
    # representing the weight at each timestep is returned
    weights = None
    x = AttentionWeightedAverage(name='attlayer', return_attention=return_attention)(x)

    if return_attention:
        x, weights = x

    if not feature_output:
        # output class probabilities
        if final_dropout_rate != 0:
            x = Dropout(final_dropout_rate)(x)

        if nb_classes > 2:
            outputs = [Dense(nb_classes, activation='softmax', name='softmax')(x)]
        else:
            outputs = [Dense(1, activation='sigmoid', name='softmax')(x)]
    else:
        # output penultimate feature vector
        outputs = [x]

    if return_attention:
        # add the attention weights to the outputs if required
        outputs.append(weights)

    return Model(inputs=[model_input], outputs=outputs)


def elsa_doc_model(hidden_dim=64,
                   dropout=0.5,
                   mode='train',
                   nb_maxlen=[20, 20],
                   nb_feature=[2248, 2248]):
    I_en = Input(shape=(nb_maxlen[0], nb_feature[1]), dtype='float32')
    en_out = AttentionWeightedAverage()(I_en)
    I_ot = Input(shape=(nb_maxlen[1], nb_feature[0]), dtype='float32')
    jp_out = AttentionWeightedAverage()(I_ot)
    O_to = concatenate([jp_out, en_out])
    O_to = Dense(hidden_dim, activation='selu')(O_to)
    if mode == 'train':
        O_to = Dropout(dropout)(O_to)
    O_out = Dense(1, activation='sigmoid', name='softmax')(O_to)
    model = Model(inputs=[I_ot, I_en], outputs=O_out)
    return model
