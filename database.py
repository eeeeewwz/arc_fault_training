import pymysql
import numpy as np
from config import DB_CONN_CONFIG


class Database():
    '''
    用于连接服务器端的MySQL数据库
    '''
    def __init__(self, host=None, user=None,
                 password=None, database=None,
                 port=3306, charset="utf8"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.charset = charset

    def db_connect(self):
        '''
        连接数据库并生成游标
        '''
        self.db = pymysql.connect(host=self.host,
                                  user=self.user,
                                  password=self.password,
                                  database=self.database,
                                  port=self.port,
                                  charset=self.charset)
        self.cursor = self.db.cursor()

        self.cursor.execute("SELECT VERSION()")
        print("MySQL Server Connected. Database version : %s" % (self.cursor.fetchone()))

    def db_disconnect(self):
        '''
        关闭数据库连接
        '''
        self.db.close()

        print("MySQL Server Disconnected.")

    def read_tables(self):
        '''
        读取数据库中的所有数据表

        Returns
        -------
        table_list : list
        '''
        sql = "SHOW TABLES"
        self.cursor.execute(sql)
        tables = self.cursor.fetchall()
        table_list = [table[0] for table in tables]

        return table_list

    def read_sample_num(self, table):
        '''
        读取指定表中的样本数量
        '''
        sql = "SELECT COUNT(*) FROM %s" % table
        self.cursor.execute(sql)

        return int(self.cursor.fetchall()[0][0])

    def read_one_sample(self, table, id):
        '''
        根据标号和样本种类读取指定表中的单个数据集样本

        Parameters
        ----------
        table : string
        id : int

        Returns
        -------
        sample_list : list 
        '''
        sql = "SELECT data FROM %s WHERE id = %d" % (table, id)
        self.cursor.execute(sql)
        sample_list = list(map(np.float32, self.cursor.fetchall()[0][0].decode().strip().split("\r\n")))

        return sample_list
    
    def read_all_samples(self, table):
        '''
        读取一个表中的所有数据样本

        Returns
        -------
        all_samples : list
        '''
        sample_num = self.read_sample_num(table)
        all_samples = []
        for idx in range(1, sample_num+1):
            all_samples.append(self.read_one_sample(table, idx))
            if idx % 500 == 0:
                print("Table %s : %d samples downloaded.\r" % (table, idx), end="")
        print("Table %s : All %d samples downloaded." % (table, sample_num))
        return all_samples

if __name__ == "__main__":
    '''
    获取表信息，以测试数据库连接
    '''
    db = Database(**DB_CONN_CONFIG)
    db.db_connect()
    table_list = db.read_tables()
    print("All tables :", table_list)

    db.db_disconnect()