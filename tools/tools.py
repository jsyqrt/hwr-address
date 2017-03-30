# coding: utf-8
# tools.py

from __future__ import division

source_script_path = '../src/'
data_file_path = '../data/'
test_script_path = '../test/'
test_sameles_path = data_file_path + 'sample/test/smallset/'

import os
import sys
sys.path.append(source_script_path)
import recognize
import math

def get_n_samples(samples_path, n):
	samples_names = os.listdir(samples_path)[:n]
	samples = {}
	for name in samples_names:
		samples[name.rstrip('.jpg')] = os.path.join(samples_path, name)
	return samples

def compare_result(name, result):
	#return levenshtein(name, result)
	length = min(len(name), len(result))
	return sum([1 if name[i]==result[i] else 0 for i in range(length)])/length

# to use levenshtein algorithm to calculate the similarity of two string.
def levenshtein(x, y):
	n = len(x)
	m = len(y)
	
	diff = [[0 for i in range(m+1)] for j in range(n+1)]
	for i in range(1, n+1):
		for j in range(1, m+1):
			if x[i-1] == y[j-1]:
				tmp = 0
			else:
				tmp = 1
			diff[i][j] = min(diff[i-1][j-1]+tmp, diff[i][j-1]+1, diff[i-1][j]+1)	
	
	sim = 1 - diff[n][m] / max(n, m)
	sim += sum([1 if x[i]==y[i] else 0 for i in range(min(m,n))])/min(m,n)
	sim /= 2
	return sim