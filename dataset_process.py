import numpy as np
from config import MODEL_CONFIG
import tensorflow as tf


def shuffle_and_slice():
    '''
    读取本地数据集，随机打乱后，
    把数据集拼起来将数据和标签对应起来，并划分训练集和测试集

    Returns
    -------
    x_train : np.ndarray
        训练集
    y_train : np.ndarray
        训练集标签
    x_test : np.ndarray
        测试集
    y_test : np.ndarray
        测试集标签
    '''
    for label, ds_name in enumerate(MODEL_CONFIG["dataset_name"]):
        file_name = "%s/%s.npy" % (MODEL_CONFIG["read_path"], ds_name)
        if label == 0:
            x_all = np.load(file_name)
            y_all = np.full(shape=x_all.shape[0],fill_value=label)
        else:
            x_one_label = np.load(file_name)
            y_one_label = np.full(shape=x_one_label.shape[0],fill_value=label)
            x_all = np.append(x_all, x_one_label, axis=0)
            y_all = np.append(y_all, y_one_label, axis=0)

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
    x = (tf.cast(x, dtype=tf.float32) - 0.8) / (1.2 - 0.8)
    y = tf.cast(y, dtype=tf.int32) # 转成整形Tensor
    y = tf.one_hot(y, depth=len(MODEL_CONFIG["dataset_name"])) # one-hot编码
    return x,y

def construct_tf_dataset():
    '''
    构造训练集，打乱、设置批大小、预处理函数

    Returns
    -------
    train_dataset : tf.data.Dataset
        训练集
    test_dataset : tf.data.Dataset
        测试集
    '''
    x_train, y_train, x_test, y_test = shuffle_and_slice()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(MODEL_CONFIG["shuffle_buffer_size"]).batch(MODEL_CONFIG["batch_size"]).map(preprocess)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(MODEL_CONFIG["batch_size"]).map(preprocess)

    return train_dataset, test_dataset

