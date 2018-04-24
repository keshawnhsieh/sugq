from __future__ import print_function
from keras.initializers import glorot_uniform
from keras.constraints import maxnorm
from keras.layers import Input
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPool1D
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
from keras import backend as K
import numpy as np
import os
import pandas as pd
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return K.sum(2*((precision*recall)/(precision+recall+1e-7))) / 4.0

def load_data(dir):
    x = np.load(dir + '/train_x.npy', mmap_mode='r')
    y = np.load(dir + '/train_y.npy', mmap_mode='r')
    vx = np.load(dir + '/eval_x.npy', mmap_mode='r')
    vy = np.load(dir + '/eval_y.npy', mmap_mode='r')

    x = np.expand_dims(x, axis=-1)
    y = np_utils.to_categorical(y, 4)
    vx = np.expand_dims(vx, axis=-1)
    vy = np_utils.to_categorical(vy, 4)

    return x, y, vx, vy

def Kenet(weight_constraint=1, dropout_rate=0.7, ks=33):
    input = Input((2600, 1))

    conv1 = Conv1D(filters=8, kernel_size=ks, strides=2, padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_constraint=maxnorm(weight_constraint),
                   )(input)

    maxp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv1)

    conv2 = Conv1D(filters=8, kernel_size=ks, strides=1, padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_constraint=maxnorm(weight_constraint),
                   )(maxp1)

    maxp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv2)

    conv3 = Conv1D(filters=16, kernel_size=ks, strides=1, padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_constraint=maxnorm(weight_constraint),
                   )(maxp2)

    maxp3 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv3)

    conv4 = Conv1D(filters=16, kernel_size=ks, strides=1, padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(seed=0),
                   kernel_constraint=maxnorm(weight_constraint),
                   )(maxp3)

    maxp4 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv4)

    flatten = Flatten()(maxp4)

    dropout1 = Dropout(dropout_rate)(flatten)

    dense1 = Dense(512, activation='relu', kernel_initializer=glorot_uniform(seed=0),
                   kernel_constraint=maxnorm(weight_constraint),
                   )(dropout1)

    dropout2 = Dropout(dropout_rate)(dense1)

    dense3 = Dense(4, activation='softmax', kernel_initializer=glorot_uniform(seed=0)
                   )(dropout2)

    model = Model(inputs=input, outputs=dense3)

    return model

def cnn_v1(ks):
    input = Input(shape=(2600, 1))

    # (2600, 1) -> (650, 4)
    conv1_1 = Conv1D(filters=4, kernel_size=ks[0], strides=2, padding='same', activation='relu')(input)
    maxp1 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv1_1)

    # (650, 4) -> (325, 8)
    conv2_1 = Conv1D(filters=8, kernel_size=ks[1], strides=1, padding='same', activation='relu')(maxp1)
    conv2_2 = Conv1D(filters=8, kernel_size=ks[1]+3, strides=1, padding='same', activation='relu')(conv2_1)
    maxp2 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv2_2)

    # (325, 8) -> (163, 16)
    conv3_1 = Conv1D(filters=16, kernel_size=ks[2], strides=1, padding='same', activation='relu')(maxp2)
    conv3_2 = Conv1D(filters=16, kernel_size=ks[2]+3, strides=1, padding='same', activation='relu')(conv3_1)
    maxp3 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv3_2)

    # (163, 16) -> (82, 16)
    conv4_1 = Conv1D(filters=16, kernel_size=ks[3], strides=1, padding='same', activation='relu')(maxp3)
    conv4_2 = Conv1D(filters=16, kernel_size=ks[3]+3, strides=1, padding='same', activation='relu')(conv4_1)
    maxp4 = MaxPool1D(pool_size=2, strides=2, padding='same')(conv4_2)

    # (82, 16) -> (82 * 16)
    flatten = Flatten()(maxp4)

    dropout1 = Dropout(0.7)(flatten)

    dense1 = Dense(512, activation='relu')(dropout1)

    dropout2 = Dropout(0.7)(dense1)

    dense2 = Dense(4, activation='softmax')(dropout2)

    model = Model(inputs=input, outputs=dense2)

    return model

def main():
    ## params
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    type = 'sec_b_fir_sugq_sec_a_sugq_tra_std'
    batch_size = 256 * 40
    epoch = 1000
    n_iter = 10
    n_particle = 5
    c1 = 0.15
    c2 = 0.25
    ## params

    # limit gpu usages
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # set_session(tf.Session(config=config))

    if not os.path.exists('model/' + type):
        os.makedirs('model/' + type)

    x, y, vx, vy = load_data('data/' + type)

    # qso optimize
    p = np.random.randint(5, 40, size=(n_particle, 4))
    # p = np.array([[23.4,  20.8,  16.6,  11.4],
    #               [26.25, 24., 20., 13.],
    #               [24.65, 21.85, 16.8, 12.3],
    #               [24.65, 20.25, 17.5, 11.75],
    #               [26.95, 24., 17.55, 13.]])

    p = np.sort(p)[:, ::-1]
    v = np.zeros([n_particle, 4])
    p_best = np.zeros([n_particle, 4])
    g_best = np.zeros([4])
    score = np.zeros([n_particle])

    summary = {'epoch': [],
               'f1': [],
               'loss': [],
               'val_f1': [],
               'val_loss': [],
               'params': [],
               'iter': [],
               'particle': [],
               }
    for it in range(n_iter):
        # one iteration
        for pa in range(n_particle):
            ks = np.round(p[pa]).astype(np.int)
            print('trying arch %s' % ks)
            model = cnn_v1(ks)

            es = EarlyStopping(monitor='val_f1', min_delta=0.0001, patience=50, verbose=1, mode='max')
            cp = ModelCheckpoint(filepath='model/%s/weights.{epoch:02d}-{val_f1:.4f}.hdf5' % type,
                                         # monitor='val_f1',
                                         verbose=1,
                                         # save_best_only=False,
                                         # mode='max'
                                         )

            callbacks_list = [es, cp]

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=[f1])

            history = model.fit(x,
                      y,
                      batch_size=batch_size,
                      epochs=epoch,
                      callbacks=callbacks_list,
                      verbose=2,
                      validation_data=(vx, vy),
                      )

            # update one particle
            pa_best = np.max(history.history['val_f1'])
            if pa_best > score[pa]:
                score[pa] = pa_best
                for i in range(4):
                    p_best[pa, i] = ks[i]
            if pa_best >= np.max(score):
                for i in range(4):
                    g_best[i] = ks[i]

            # export history
            tmp = history.history
            tmp['params'] = [ks] * len(history.epoch)
            tmp['epoch'] = history.epoch
            tmp['particle'] = [pa] * len(history.epoch)
            tmp['iter'] = [it] * len(history.epoch)

            _ = {k: summary[k].extend(tmp[k]) for k in summary}
            del model

        print('iter %d finished' % it)
        print('p_best:\n', p_best)
        print('g_best:\n', g_best)
        print('score:\n', score)

        # update particle swarm
        v = v + c1 * (p_best - p) + c2 * (g_best - p)
        p = p + v

        print('update arch')
        print('new arch:\n', p)

    df = pd.DataFrame.from_dict(summary)
    from datetime import datetime
    df.to_csv('diagnosis/summary_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S'))

if __name__ == '__main__':
    main()