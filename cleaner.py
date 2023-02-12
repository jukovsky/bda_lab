import csv
import os
import sys
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

COLLECTION_NAME = 'clean_df'
DB_NAME = 'bda_project'
REDDIT_NAME = 'Combined_News_DJIA.csv'
DJIA_NAME = 'upload_DJIA_table.csv'
OIL_NAME = 'yahoofinance-oildata.csv'


def run():
    mongoClient = MongoClient()
    csvfile = open('data/' + REDDIT_NAME, 'r')
    reader = csv.DictReader(csvfile)
    db = mongoClient.get_database(DB_NAME)
    collection = db[COLLECTION_NAME]
    collection.drop()

    for row in reader:
        collection.insert_one(row)

    collection.update_many({}, {"$rename": {"Label": "DJIA_LABEL"}, "$set": {"DJIA_CLOSE": 0 , "DJIA_LABEL_CORRECT": 0, "OIL_CLOSE": 0, "OIL_LABEL_CORRECT": 0}})

    csvfile = open('data/' + DJIA_NAME, 'r')
    reader = csv.DictReader(csvfile)
    prev = 0
    for row in reader:
        close = round(float(row["Close"]), 2)
        label = 1 if close > prev else 0
        prev = close
        collection.update_one({"Date": row["Date"]}, {"$set": {"DJIA_CLOSE": close, "DJIA_LABEL_CORRECT": label}})

    csvfile = open('data/' + OIL_NAME, 'r')
    reader = csv.DictReader(csvfile)
    prev = 0
    for row in reader:
        try:
            close = round(float(row["Close*"]), 2)
        except ValueError:
            continue
        label = 1 if close > prev else 0
        prev = close
        formattedDate = datetime.strptime(row["Date"], '%b %d, %Y').strftime('%Y-%m-%d')
        collection.update_one({"Date": formattedDate}, {"$set": {"OIL_CLOSE": row["Close*"], "OIL_LABEL_CORRECT": label}})

    collection.update_many({}, {"$unset": {"DJIA_LABEL": 1}})
    collection.update_many({}, {"$rename": {"DJIA_LABEL_CORRECT": "DJIA_LABEL", "OIL_LABEL_CORRECT": "OIL_LABEL"}})

    tops = []
    for i in range(1, 26):
        tops.append({ "$convert": { "input": "$Top" + str(i), "to": "string", "onNull": "" } })
        tops.append(" ")
        pipeline = [
            {
                "$set": {
                    "Top" + str(i): {
                        "$replaceAll": {
                            "input": "$Top" + str(i),
                            "find": 'b"',
                            "replacement": ""
                        }
                    }
                }
            },
            {
                "$set": {
                    "Top" + str(i): {
                        "$replaceAll": {
                            "input": "$Top" + str(i),
                            "find": 'b\'',
                            "replacement": ""
                        }
                    }
                }
            },
            {
                "$set": {
                    "Top" + str(i): {
                        "$replaceAll": {
                            "input": "$Top" + str(i),
                            "find": '\\\\',
                            "replacement": ""
                        }
                    }
                }
            },
            {
                "$set": {
                    "Top" + str(i): {
                        "$replaceAll": {
                            "input": "$Top" + str(i),
                            "find": '\\"',
                            "replacement": ""
                        }
                    }
                }
            },
            {
                "$set": {
                    "Top" + str(i): {
                        "$replaceAll": {
                            "input": "$Top" + str(i),
                            "find": '"',
                            "replacement": ""
                        }
                    }
                }
            }
        ]
        collection.update_many({}, pipeline)

    collection.update_many({}, [{"$set": {"News": {"$concat": tops}}}])

    for i in range(1, 26):
        collection.update_many({}, {"$unset": {"Top" + str(i): 1}})

    data = collection.find()
    print(pd.DataFrame(data))

    print('Import finished')
