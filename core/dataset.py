import math
from pathlib import Path

import numpy as np
import tensorflow as tf
from functools import partial

from sklearn.model_selection import train_test_split


def make_epic_seqs(file_name, win_size, data_rate=1000, anno_rate=20, split=0, n_rows=0):
    """

    :param file_name: full path to physiology data
    :param win_size:  window size in second
    :param data_rate: recorded data rate
    :param anno_rate: annotation rate
    :param split: 0 for train and 1 for test
    :return:
    """
    cur_expr_name = tf.strings.split(file_name, '/')[-1]
    window_size = int(win_size * data_rate)
    shift = data_rate // anno_rate

    nskip = (10 * data_rate - window_size) * split + 1
    nskip_anno = math.ceil(window_size / shift) * (1 - split)

    phys_file = tf.data.experimental.CsvDataset(file_name, record_defaults=[0., ] * 9, header=True).skip(nskip).map(
        lambda *x: tf.stack(x)).window(size=window_size, shift=shift, drop_remainder=True).flat_map(
        lambda x: x.batch(window_size)).map(lambda phys: {'phys': phys, 'name': cur_expr_name})
    anno_file = tf.data.experimental.CsvDataset(tf.strings.regex_replace(file_name, 'physiology', 'annotations'),
                                                record_defaults=[0.] * 3,
                                                header=True).skip(nskip_anno).map(
        lambda *x: {'anno': tf.stack(x), 'name': cur_expr_name})

    zip_dataset = tf.data.Dataset.zip((phys_file, anno_file))
    if split == 0:
        return zip_dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
    else:
        return zip_dataset


def epic_dataloader(data_dir: str = "/home/hvthong/sXProject/EPiC23/data", scenario: int = 1, fold: int = 0,
                    win_size=2, val_ratio: float = 0., batch_size=256):
    if scenario == 1:
        data_path = Path(data_dir) / f'scenario_{scenario}'

    else:
        data_path = Path(data_dir) / f'scenario_{scenario}/fold_{fold}'

    train_files = sorted([x.__str__() for x in data_path.glob("train/physiology/*.csv")])
    test_files = sorted([x.__str__() for x in data_path.glob("test/physiology/*.csv")])

    if scenario == 2 and val_ratio > 0.:
        # Do split train_files into train and val based on subject ID
        participants = np.unique([x.split('/')[-1].split('_')[1] for x in train_files])
        train_part, val_part = train_test_split(participants, test_size=val_ratio,
                                                random_state=len(participants), shuffle=True)
        print('Training subjects: ', train_part)
        print('Validation subjects: ', val_part)

        train_part_split = [x for x in train_files if x.split('/')[-1].split('_')[1] in train_part]
        val_part_split = [x for x in train_files if x.split('/')[-1].split('_')[1] in val_part]

        train_files = sorted(train_part_split)
        val_files = sorted(val_part_split)
    else:
        val_files = None

    data_rate = 1000
    anno_rate = 20
    cycle_length = 16
    autotune = tf.data.AUTOTUNE
    make_epic_seqs_train = partial(make_epic_seqs, win_size=win_size, data_rate=data_rate, anno_rate=anno_rate, split=0)
    make_epic_seqs_test = partial(make_epic_seqs, win_size=win_size, data_rate=data_rate, anno_rate=anno_rate, split=1)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_files).cache().shuffle(300,
                                                                                    reshuffle_each_iteration=True).interleave(
        make_epic_seqs_train, cycle_length=cycle_length, num_parallel_calls=autotune).shuffle(1024).repeat()

    test_dataset = tf.data.Dataset.from_tensor_slices(test_files).cache().interleave(make_epic_seqs_test,
                                                                                     cycle_length=cycle_length,
                                                                                     num_parallel_calls=autotune)

    data_loader = {'train': train_dataset.prefetch(autotune).batch(batch_size=batch_size, num_parallel_calls=autotune),
                   'val': None,
                   'test': test_dataset.prefetch(autotune).batch(batch_size=batch_size, num_parallel_calls=autotune)}

    if scenario == 2 and val_files is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices(val_files).cache().interleave(make_epic_seqs_test,
                                                                                       cycle_length=cycle_length,
                                                                                       num_parallel_calls=autotune)
        data_loader['val'] = val_dataset.prefetch(autotune).batch(batch_size=batch_size, num_parallel_calls=autotune)

    return data_loader
