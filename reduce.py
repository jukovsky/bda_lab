from pymongo import MongoClient
from tabulate import tabulate
import pandas as pd

COLLECTION_NAME = 'clean_df'
DB_NAME = 'bda_project'

def run():
    mongoClient = MongoClient()

    db = mongoClient.get_database(DB_NAME)
    collection = db[COLLECTION_NAME]
    data = collection.find({"News": {"$regex": 'war ', "$options" :'i'}, "OIL_LABEL": 1})
    oilOne = len(pd.DataFrame(data))
    print(pd.DataFrame(data))

    data = collection.find({"News": {"$regex": 'war ', "$options" :'i'}, "OIL_LABEL": 0})
    oilZero = len(pd.DataFrame(data))
    print(pd.DataFrame(data))

    data = collection.find({"News": {"$regex": 'war ', "$options" :'i'}, "DJIA_LABEL": 1})
    djiaOne = len(pd.DataFrame(data))
    print(pd.DataFrame(data))

    data = collection.find({"News": {"$regex": 'war ', "$options" :'i'}, "DJIA_LABEL": 0})
    djiaZero = len(pd.DataFrame(data))
    print(pd.DataFrame(data))

    table = [
        ['OIL_LABEL_1', oilOne],
        ['OIL_LABEL_0', oilZero],
        ['DJIA_LABEL_1', djiaOne],
        ['DJIA_LABEL_0', djiaZero],
    ]
    print(tabulate(table))