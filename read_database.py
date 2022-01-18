import pymysql


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
        数据库连接
        '''
        self.db = pymysql.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    port=self.port,
                    charset=self.charset)
        self.cursor = self.db.cursor()


    def db_disconnect(self):
        '''
        关闭数据库连接
        '''
        self.db.close()

    def get_data(self, table, key, key_value):
        '''
        根据标号和样本类别读取数据集样本
        table : 样本种类表
        key : 
        key_value :
        '''
        