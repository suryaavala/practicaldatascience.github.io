#!/Users/duyicong/anaconda/envs/MRtutorial/bin/python
import sys

#reading line from standard input
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        #write result to standard output
        print ('{0}\t{1}'.format(word, 1))
