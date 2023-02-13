import argparse
import cleaner
import forest
import logreg
import reduce

parser = argparse.ArgumentParser()
parser.add_argument('command', nargs = '?')
args = parser.parse_args()

if args.command is None:
    f = open('help.txt', 'r')
    helpText = f.read()
    print(helpText)

if args.command == 'import':
    cleaner.run()

if args.command == 'logreg':
    logreg.run()

if args.command == 'forest':
    forest.run()

if args.command == 'reduce':
    reduce.run()
