import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras import layers, losses, optimizers
from config import MODEL_CONFIG
from dataset_process import construct_tf_dataset


print("TensorFlow Version :", tf.__version__)
print("GPU Ability :", tf.test.is_gpu_available())

train_ds = construct_tf_dataset()
sample_point = 800

# 建立模型
model = Sequential([
    layers.Conv1D(6, kernel_size=3, strides=1), # 第一个卷积层，6个长度为3的卷积核
    layers.MaxPooling1D(pool_size=2, strides=2), # 减半的池化层
    layers.ReLU(), # 激活函数
    layers.Conv1D(16, kernel_size=3, strides=1), # 第二个卷积层，16个长度为3的卷积核
    layers.MaxPooling1D(pool_size=2, strides=2), # 减半的池化层
    layers.ReLU(), # 激活函数
    layers.Flatten(), # 打平层，方便全连接层处理
    layers.Dense(120, activation='relu'), # 全连接层，100个节点
    layers.Dense(2) # 全连接层，2个节点
    ])
model.build(input_shape=([None, sample_point]))
model.summary()

# from_logits=True，可以把softmax交给系统去算，避免数值溢出，提高稳定性
criterion = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"])

for epoch in range(MODEL_CONFIG["epoch"]):
    for batch_step, (x_train,y_train) in enumerate(train_ds): # 迭代 Step 数
        with tf.GradientTape() as tape:
            # 插入通道维度， [b,800] =>[b,800,1]
            x_train = tf.reshape(x_train, [-1, sample_point])
            # 前向计算，获得0-1分类的概率分布 => [b, 2]
            out = model(x_train)
            # 计算交叉熵损失函数
            loss = criterion(y_train, out)
        # 自动计算梯度
        grads = tape.gradient(loss, model.trainable_variables)
        # 自动更新参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 训练完成后保存网络模型和参数
model.save('TEST_MODEL')