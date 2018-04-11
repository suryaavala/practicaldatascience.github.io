import csv
import matplotlib.pyplot as plt
import numpy as np

with open('pathology.tsv') as tsvfile:
	reader = csv.reader(tsvfile, delimiter='\t')
	row_count = 0
	all_genes = set()
	highs = dict()
	meds = dict()
	lows = dict()
	for row in reader:
		high = row[3]
		medium = row[4]
		low = row[5]
		if (high != 'High' and high != ''):
			row_count += 1
			all_genes.add(row[0])
			target = row[2]
			cur_high = highs.get(target)
			cur_med = meds.get(target)
			cur_low = lows.get(target)
			
			if (cur_high == None):
				highs[target] = int(high)
			else:
				highs[target] += int(high)

			if (cur_med == None):
				meds[target] = int(medium)
			else:
				meds[target] += int(medium)

			if (cur_low == None):
				lows[target] = int(low)
			else:
				lows[target] += int(low)

	print("Number of Rows: ", row_count)
	print("Number of Genes:", len(all_genes))

	names = highs.keys()
	high_vals = highs.values()
	med_vals = meds.values()
	low_vals = lows.values()

	plt.bar(names, high_vals)
	plt.xticks(rotation='vertical')
	plt.show
	