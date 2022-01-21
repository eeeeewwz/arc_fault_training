# Database Connect Configuration
DB_CONN_CONFIG = {
    "host"      : "10.99.110.187",
    "user"      : "root",
    "password"  : "123456",
    "database"  : "arc_database",
    "port"      : 3306,
    "charset"   : "utf8"
}

# Database Fetch Data Configuration
FETCH_DATA_CONFIG = {
    "save_path"     : "dataset",
    "dataset_name"  : ["hongganji", "bulb_arc", "bulb_normal",
                       "xianshiqi_arc", "xianshiqi_normal",
                       "xichenqi_arc", "xichenqi_normal"],
    "fetch_all"     : False,
    # 还未支持修改
    "shuffle"       : False,
    "data_type"     : "float32"
}

# TensorFlow Model Configuration
MODEL_CONFIG = {
    # 数据集配置参数
    "read_path"     : "dataset",
    "dataset_name"  : ["bulb_arc", "bulb_normal",
                       "xichenqi_arc", "xichenqi_normal",
                       "xianshiqi_arc", "xianshiqi_normal"],
    "training_ratio": 0.8,
    "shuffle_buffer_size"   : 30000,    # 请尽量让它大于所有参与训练样本的总和，这样shuffle能够更彻底
    # 模型架构参数

    # 模型训练超参数
    "learning_rate" : 0.001,
    "batch_size"    : 500,
    "epoch"         : 50,
    # 模型存储参数
    "model_path"    : "models",
    "model_name"    : "xianshiqi"
}