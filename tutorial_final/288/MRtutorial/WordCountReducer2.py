#!/usr/bin/env python
import sys
curr_word = None
curr_count = 0
word_count_dict = {}
# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    word, count = line.split('\t', 1)

    try:
        count = int(count)
    except ValueError:
        continue

    if not curr_word:
        curr_word = word

    if word == curr_word:
        curr_count += count
    else:

        print('{0}\t{1}'.format(curr_word, curr_count))
        curr_word = word
        curr_count = count

#last can be combined with previous ones
if curr_word == word:
    print('{0}\t{1}'.format(curr_word, curr_count))










