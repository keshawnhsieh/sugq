import pandas as pd
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
from config import *

class BatchDataset:
    base = {key: [] for key in range(n_classes)}
    train = []
    eval = []
    test = []
    add1 = {key: [] for key in range(n_classes)}
    add2 = {key: [] for key in range(n_classes)}
    ## params, use samples labeled in addx_filter
    add1_set = True
    add2_set = True
    add1_filter = [0, 1, 2, 3]
    add2_filter = [0, 1, 2, 3]
    ## params

    def __init__(self, eval_size=0.2):
        # base dataset
        csv = pd.read_csv(base_csv)
        id, label = list(csv['id']), list(csv['type'])

        for _id, _label in zip(id, label):
            self.base[label_map[_label]].append(base_dir + '/' + str(_id) + '.txt')

        for k in self.base:
            k_len = len(self.base[k])
            eval_len = int(k_len * eval_size)
            eval_idx = random.sample(range(k_len), eval_len)
            train_idx = set(range(k_len)) - set(eval_idx)
            train_idx = list(train_idx)

            self.train.extend([[self.base[k][ii], k] for ii in train_idx])
            self.eval.extend([[self.base[k][ii], k] for ii in eval_idx])

        # add ext1 dataset
        if self.add1_set:
            csv = pd.read_csv(add1_csv)
            id, label = list(csv['id']), list(csv['type'])

            for _id, _label in zip(id, label):
                self.add1[label_map[_label]].append(add1_dir + '/' + str(_id) + '.txt')

            for k in self.add1_filter:
                self.train.extend([[ii, k] for ii in self.add1[k]])

        # add ext2 dataset
        if self.add2_set:
            csv = pd.read_csv(add2_csv)
            id, label = list(csv['id']), list(csv['type'])

            for _id, _label in zip(id, label):
                self.add2[label_map[_label]].append(add2_dir + '/' + str(_id) + '.txt')

            for k in self.add2_filter:
                self.train.extend([[ii, k] for ii in self.add2[k]])

        # shuffle
        self.train, self.eval = np.array(self.train), np.array(self.eval)

        perm = np.arange(self.train.shape[0])
        np.random.shuffle(perm)
        self.train = self.train[perm]

        perm = np.arange(self.eval.shape[0])
        np.random.shuffle(perm)
        self.eval = self.eval[perm]

    def get_test(self):
        csv = pd.read_csv(test_csv)
        id = list(csv['id'])

        self.test.extend([test_dir + '/' + str(ii) + '.txt'] for ii in id)

        return np.array(self.test)

if __name__ == '__main__':
    reader = BatchDataset(eval_size=0.1)
    test_ds = reader.get_test()
    pass

