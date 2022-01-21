from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, layers, losses, metrics
from config import MODEL_CONFIG
from dataset_process import construct_tf_dataset


print("TensorFlow Version :", tf.__version__)
print("GPU Ability :", tf.test.is_gpu_available())

train_dataset, test_dataset = construct_tf_dataset()
sample_point = next(iter(train_dataset))[0].shape[1]
catgory_num = len(MODEL_CONFIG["dataset_name"])

# 建立模型
model = Sequential([
    layers.Conv1D(filters=10, kernel_size=10, strides=1), # 第一个卷积层，有<filter>个长度为<kernel_size>的卷积核，步长为<strides>
    layers.MaxPooling1D(pool_size=2, strides=2), # 减半的池化层，池化窗口为2，池化步长为2
    layers.ReLU(), # ReLU激活函数
    layers.Conv1D(filters=10, kernel_size=5, strides=1), # 第二个卷积层
    layers.MaxPooling1D(pool_size=2, strides=2), # 减半的池化层
    layers.ReLU(), # ReLU激活函数
    layers.Flatten(), # 展平层，方便全连接层处理
    layers.Dense(150, activation='relu'), # 展平后的全连接层
    layers.Dense(catgory_num) # 输出全连接层，节点数为分类数量<catgory_num>
    ])
model.build(input_shape=([None, sample_point, 1]))
model.summary()

# from_logits=True，可以把softmax交给系统去算，避免数值溢出，提高稳定性
criterion = losses.CategoricalCrossentropy(from_logits=True)
optimizer = optimizers.Adam(learning_rate=MODEL_CONFIG["learning_rate"])

# 跟踪的损失函数值和准确率
train_loss_results = []
train_accuracy_results = []
test_loss_results = []
test_accuracy_results = []

train_loss = metrics.Mean("train_loss", dtype=tf.float32)
train_accuracy = metrics.CategoricalAccuracy("train_accuracy")
test_loss = metrics.Mean("test_loss", dtype=tf.float32)
test_accuracy = metrics.CategoricalAccuracy("test_accuracy")

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "logs/gradient_tape/" + current_time + "/train"
test_log_dir = "logs/gradient_tape/" + current_time + "/test"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 模型训练
for epoch in range(MODEL_CONFIG["epoch"]):
    # 每个epoch以batch为单位进行训练
    for batch_step, (x_train, y_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x_train = tf.expand_dims(x_train, axis=2)
            # 前向传播，获得分类的概率分布
            y_ = model(x_train, training=True)
            # 计算交叉熵损失函数
            loss = criterion(y_true=y_train, y_pred=y_)
        # 计算梯度并进行反向传播更新参数
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 在metrics中更新训练集损失函数值和准确率
        train_loss.update_state(loss)
        train_accuracy.update_state(y_train, y_)

        if batch_step % 50 == 0:
            print("Epoch {:d} Step {:d}, Average Train Loss {:f},"
                  "Train Accuracy {:f}".format(epoch,
                                               batch_step,
                                               train_loss.result(),
                                               train_accuracy.result()))

    # 记录训练集上的损失函数值和准确率，写入logs中，供TensorBoard显示
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", train_loss.result(), step=epoch)
        tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

    train_loss_results.append(train_loss.result())
    train_accuracy_results.append(train_accuracy.result())

    # 每个epoch完成后跑一遍测试集看准确率
    for (x_test, y_test) in test_dataset:
        y_ = model(x_test)
        loss = criterion(y_true=y_test, y_pred=y_)

        # 在metrics中更新测试集损失函数值和准确率
        test_loss.update_state(loss)
        test_accuracy.update_state(y_test, y_)

    # 记录测试集上的损失函数值和准确率，写入logs中，供TensorBoard显示
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    test_loss_results.append(test_loss.result())
    test_accuracy_results.append(test_accuracy.result())

    print("Epoch {:d} done, Train Loss {:f}, Train Accuracy {:f},"
          "Test Loss {:f}, Test Accuracy {:f}".format(epoch,
                                                      train_loss.result(),
                                                      train_accuracy.result(),
                                                      test_loss.result(),
                                                      test_accuracy.result()))

    # 每个epoch结束后重置metrics
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

# 训练完成后保存网络模型和参数
model.save(MODEL_CONFIG["model_path"] + "/" + MODEL_CONFIG["model_name"])