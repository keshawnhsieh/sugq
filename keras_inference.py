from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from keras import backend as K
from config import *

## params
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
test_dir = 'data/sec_b_fir_sugq_sec_a_sugq_tra_std'
model_file = [
        # sec_b > 0.810
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.171-0.8141.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.180-0.8111.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.165-0.8156.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.150-0.8131.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.153-0.8142.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.196-0.8134.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.202-0.8114.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.220-0.8127.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.236-0.8121.hdf5',
        # sec_b > 0.809
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.175-0.8097.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.162-0.8096.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.184-0.8093.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.168-0.8093.hdf5',
        'model/sec_b_fir_sugq_sec_a_sugq_tra_std/weights.215-0.8098.hdf5',

        # # sec_a > 0.830
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.165-0.8316.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.166-0.8303.hdf5',
        # # sec_a > 0.829
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.134-0.8291.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.136-0.8298.hdf5',
        # # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.199-0.8298.hdf5',
        # # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.201-0.8297.hdf5',
        # # sec_a > 0.828
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.135-0.8286.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.153-0.8282.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.157-0.8282.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.163-0.8286.hdf5',
        # # sec_a > 0.827
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.142-0.8275.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.172-0.8270.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.175-0.8273.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.213-0.8270.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.171-0.8273.hdf5',
        # 'model/sec_a_fir_sugq_sec_b_sugq_tra_std/weights.184-0.8275.hdf5',
            ]
## params

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

    return K.sum(2 * ((precision * recall) / (precision + recall + 1e-7))) / 4.0

def most_common_element(array):
    (values, counts) = np.unique(array, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

def load_test(dir):
    test_x = np.load(dir + '/test_x.npy')
    test_x = np.expand_dims(test_x, axis=-1)

    return test_x

def main():
    test_x = load_test(test_dir)

    df = pd.read_csv(test_csv).as_matrix()

    test_y = []
    for mf in tqdm(model_file):
        model = load_model(mf, custom_objects={'f1': f1})

        y = model.predict(test_x, batch_size=512, verbose=0)
        test_y.append(y)

        # save logits
        # csv_logits = np.hstack((df, y))
        # csv_logits = pd.DataFrame(csv_logits)
        # csv_logits.to_csv('result/' + mf.split('/')[-1][:-5] + '.csv',
        #                   header=['id', 'star', 'unknown', 'galaxy', 'qso'],
        #                   index=False)

    sum_y = np.sum(np.array(test_y), axis=0)

    vote_y_idx = np.argmax(sum_y, axis=-1)
    vote_y = [label_map_inv[e] for e in vote_y_idx]

    # save merged logits
    # mean_y = sum_y / len(model_file)
    # merge_y = np.hstack((df, mean_y, np.expand_dims(vote_y, axis=-1)))
    # pd.DataFrame(merge_y).to_csv('result/pred_lgt.csv', header=['id', 'star', 'unknown', 'galaxy', 'qso', 'pred'], index=False)

    # generate csv result
    csv_content = np.hstack((df, np.expand_dims(vote_y, axis=-1)))

    csv_content = pd.DataFrame(csv_content)
    from datetime import datetime
    name = datetime.now().strftime('submit/submit_%Y%m%d_%H%M%S') + '.csv'
    csv_content.to_csv(name, header=False, index=False)
    print('before correction')
    print csv_content[1].value_counts()

if __name__ == '__main__':
    main()






