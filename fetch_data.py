import numpy as np
from os.path import exists
from database import Database
from config import DB_CONN_CONFIG, FETCH_DATA_CONFIG


def fetch_data():
    '''
    从MySQL Server下载指定数据集到本地，以Numpy npy格式保存

    Parameters
    ----------

    Returns
    -------
    files : list
        本次下载的所有的数据集文件名
    '''
    db = Database(**DB_CONN_CONFIG)
    db.db_connect()

    tables = db.read_tables()
    print("ALL DATASETS:")
    for idx in range(len(tables)):
        print("Dataset %d : %s" % (idx, tables[idx]))
    print("-" * 50)

    print("Fetch dataset from MySQL Server...")
    # 若配置中写了fetch_all属性为True，则把所有数据集存到本地
    tbl_sel = tables if FETCH_DATA_CONFIG["fetch_all"] else FETCH_DATA_CONFIG["dataset_name"]
    files = []
    for table in tbl_sel:
        file_name = "%s/%s.npy" % (FETCH_DATA_CONFIG["save_path"], table)
        # 首先检查数据集是否已存在本地
        if exists(file_name):
            print("Dataset file %s already exists. Skip this." % (file_name))
        else:
            data = db.read_all_samples(table)
            np.save(file_name, data)
            del data
            files.append(file_name)

    db.db_disconnect()
    return files

if __name__ == "__main__":
    '''
    这里Fetch Data和Construct Model是完全独立的，
    你也可以选择在配置完FETCH_DATA_CONFIG后单独运行这个fetch_data.py，先把数据集fetch到本地
    '''
    fetch_data()