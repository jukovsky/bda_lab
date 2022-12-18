import argparse
import csv
import forest
import logreg
import os
import sys

from pymongo import MongoClient

parser = argparse.ArgumentParser()
parser.add_argument('command', nargs = '?')
args = parser.parse_args()

if args.command is None:
    f = open('help.txt', 'r')
    helpText = f.read()
    print(helpText)

if args.command == 'import':
    mongoClient = MongoClient()

    for file in os.listdir('data'):
        filename = os.path.splitext(file)
        db = mongoClient.get_database(filename[0])
        db.segment.drop()

        csvfile = open('data/' + file, 'r')
        reader = csv.DictReader(csvfile)

        for each in reader:
            db.segment.insert_one(each)

        data = db.segment.find()
        print(pd.DataFrame(data))

    print('Import finished')

if args.command == 'logreg':
    logreg.run()

if args.command == 'forest':
    forest.run()
