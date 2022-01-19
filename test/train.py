import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, losses, optimizers
print(tf.__version__)
print(tf.test.is_gpu_available())

# 读取MNIST数据集
(x,y),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

def preprocess(x, y): # 自定义的预处理函数
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32) # 转成整形张量
    y = tf.one_hot(y, depth=10) # one-hot编码
    return x,y

# 构造训练集，打乱、设置批大小、预处理函数以及重复次数(epoch)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(buffer_size = 10000)
train_db = train_db.batch(batch_size = 512)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(3)

# 定义LeNet-5
network = Sequential([ # 网络容器
    layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
    layers.ReLU(), # 激活函数
    layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16 个 3x3 卷积核
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
    layers.ReLU(), # 激活函数
    layers.Flatten(), # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'), # 全连接层， 120 个节点
    layers.Dense(84, activation='relu'), # 全连接层， 84 节点
    layers.Dense(10) # 全连接层， 10 个节点
    ])
network.build(input_shape=(None, 28, 28,1))
network.summary()

# from_logits=True，可以把softmax交给系统去算，避免数值溢出，提高稳定性
criteon = losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.RMSprop(0.001)

for step, (x,y) in enumerate(train_db): # 迭代 Step 数
    with tf.GradientTape() as tape:
        # 插入通道维度， [b,28,28] =>[b,28,28,1]
        x = tf.expand_dims(x,axis=3)
        # 前向计算，获得 10 类别的概率分布 => [b, 10]
        out = network(x)
        # 计算交叉熵损失函数
        loss = criteon(y, out)
    # 自动计算梯度
    grads = tape.gradient(loss, network.trainable_variables)
    # 自动更新参数
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

# 训练完成后保存网络模型和参数
network.save('LeNet-5.h5')