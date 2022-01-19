import numpy as np
from database import Database
from config import DB_CONN_CONFIG, FETCH_DATA_CONFIG

def save_npy():
    db = Database(**DB_CONN_CONFIG)
    db.db_connect()
    
    db.db_disconnect()

def load_data():
    pass