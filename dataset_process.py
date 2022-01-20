import numpy as np
from config import MODEL_CONFIG
import tensorflow as tf


def shuffle_and_slice():
    '''
    读取本地数据集，随机打乱后，
    把数据集拼起来将数据和标签对应起来，并划分训练集和测试集
    '''
    x_all, y_all = [], []
    for label, ds_name in enumerate(MODEL_CONFIG["dataset_name"]):
        file_name = "%s/%s.npy" % (MODEL_CONFIG["read_path"], ds_name)
        x_one_label = np.load(file_name)
        y_one_label = np.full(shape=x_one_label.shape[0],fill_value=label)
        np.append(x_all, x_one_label, axis=0)
        np.append(y_all, y_one_label, axis=0)

    permutation = np.arange(len(x_all))
    np.random.shuffle(permutation)
    x_all = x_all[permutation]
    y_all = y_all[permutation]

    slice_num = int(MODEL_CONFIG["training_ratio"] * len(x_all))
    x_train, y_train = x_all[ :slice_num], y_all[ :slice_num]
    x_test, y_test = x_all[slice_num: ], y_all[slice_num: ]
    
    return x_train, y_train, x_test, y_test

def preprocess(x, y):
    '''
    数据集预处理，完成归一化和one-hot编码
    '''
    x = tf.cast(x, dtype=tf.float32) / 2.
    y = tf.cast(y, dtype=tf.int8) # 转成整形张量
    y = tf.one_hot(y, depth=2) # one-hot编码
    return x,y

def construct_tf_dataset():
    '''
    构造训练集，打乱、设置批大小、预处理函数以及重复次数(epoch)
    '''
    x_train, y_train, _, _ = shuffle_and_slice()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size = MODEL_CONFIG["shuffle_buffer_size"])
    train_ds = train_ds.batch(batch_size = MODEL_CONFIG["batch_size"])
    train_ds = train_ds.map(preprocess)

    return train_ds

