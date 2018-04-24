from batch_data_reader import BatchDataset
import numpy as np
from tqdm import tqdm
import os

def parse_line(file):
    with open(file, 'r') as f:
        content = f.readline().rstrip('\n').split(',')

    content = map(np.float32, content)
    # content -= np.mean(content)
    # content /= np.std(content)
    content /= np.mean(content)

    return np.array(content)

def parse_txt(array):
    parsed = []
    for sample in tqdm(array):
        if sample.shape[0] > 1:
            parsed.append(np.hstack((parse_line(sample[0]), sample[None, 1].astype(np.int64))))
        else:
            parsed.append(parse_line(sample[0]))

    parsed = np.array(parsed)
    if array[0].shape[0] > 1:
        return parsed[:, :2600].astype(np.float32), parsed[:, 2600].astype(np.int32)
    else:
        return parsed.astype(np.float32)

if __name__ == '__main__':
    ## params
    dir = 'data/sec_b_fir_sugq_sec_a_sugq_tra_std'
    ## params

    if not os.path.exists(dir):
        os.makedirs(dir)

    reader = BatchDataset(eval_size=0.1)

    print('writing train_x.npy and train_y.npy...')
    train_x, train_y = parse_txt(reader.train)
    np.save(dir + '/train_x.npy', train_x)
    np.save(dir + '/train_y.npy', train_y)

    print('writing eval_x.npy and eval_y.npy...')
    eval_x, eval_y = parse_txt(reader.eval)
    np.save(dir + '/eval_x.npy', eval_x)
    np.save(dir + '/eval_y.npy', eval_y)

    print('writing test_x.npy...')
    test_x = parse_txt(reader.get_test())
    np.save(dir + '/test_x.npy', test_x)




