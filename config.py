first_dir = 'data/first_train_data_20180131'
first_csv = 'data/first_train_index_20180131.csv'

second_a_dir = 'data/second_a_train_data_20180313'
second_a_csv = 'data/second_a_train_index_20180313.csv'

second_b_dir = 'data/second_b_train_data_20180313'
second_b_csv = 'data/second_b_train_index_20180313.csv'

test_dir = 'data/final_rank_data_20180413'
test_csv = 'data/final_rank_index_20180413.csv'

n_classes = 4

label_map = {'qso': 3, 'unknown': 1, 'star': 0, 'galaxy': 2}
label_map_inv = dict((k, v) for v, k in label_map.items())

base_dir = second_b_dir
base_csv = second_b_csv

add1_dir = first_dir
add1_csv = first_csv

add2_dir = second_a_dir
add2_csv = second_a_csv