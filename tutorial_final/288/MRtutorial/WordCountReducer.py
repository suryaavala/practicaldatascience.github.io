#!/usr/bin/env python
import sys
word_count_dict = {}
# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue

    if word in word_count_dict:
        word_count_dict[word] += count
    else:
        word_count_dict[word] = count

for word in word_count_dict.keys():
    print('{0}\t{1}'.format(word, word_count_dict[word]))


