import tensorflow as tf
print(tf.__version__)
print(tf.test.is_gpu_available())

# 读取MNIST数据集
(x,y),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

def preprocess2(x, y): # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32) # 转成整形张量
    return x,y

# 构造训练集，打乱、设置批大小、预处理函数
test_db = tf.data.Dataset.from_tensor_slices((x, y))
test_db = test_db.shuffle(buffer_size = 10000)
test_db = test_db.batch(batch_size = 512)
test_db = test_db.map(preprocess2)

# 恢复LeNet-5
network = tf.keras.models.load_model('LeNet-5.h5')

# 记录预测正确的数量，总样本数量
correct, total = 0,0
for x,y in test_db: # 遍历所有训练集样本
    # 插入通道维度， =>[b,28,28,1]
    x = tf.expand_dims(x,axis=3)
    # 前向计算，获得10类别的预测分布， [b, 784] => [b, 10]
    out = network(x)
    pred = tf.argmax(out, axis=-1)
    y = tf.cast(y, tf.int64)
    # 统计预测正确数量
    correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.float32)))
    # 统计预测样本总数
    total += x.shape[0]
# 计算准确率
print('test acc:', correct/total)