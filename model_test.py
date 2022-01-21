import tensorflow as tf
from tensorflow.keras import models
from config import DB_CONN_CONFIG, FETCH_DATA_CONFIG, MODEL_CONFIG
from matplotlib import pyplot
from database import Database
import numpy as np

# 按种类对模型用到的数据集进行测试
model = models.load_model(MODEL_CONFIG["model_path"] + "/" + MODEL_CONFIG["model_name"])
all_sample = {}


for idx, ds_name in enumerate(MODEL_CONFIG["dataset_name"]):
    raw_data = np.load(MODEL_CONFIG["read_path"] + "/" + ds_name + ".npy")
    fixed_data = (raw_data - 0.8) / (1.2 - 0.8)
    all_sample[ds_name] = tf.expand_dims(fixed_data, axis=2)

    pred = tf.argmax(model(all_sample[ds_name]), axis=-1)
    y = np.full(shape=pred.shape, fill_value=idx)
    correct_num = int(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.float32)))
    accuracy = correct_num / len(all_sample[ds_name])

    print("Dataset: {}, {}/{} test case validated, test Accuracy {}".format(ds_name,
                                                                            correct_num,
                                                                            len(all_sample[ds_name]),
                                                                            accuracy))
