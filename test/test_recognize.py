# coding: utf-8
# test_recognize.py to test `../src/recognize.py`
from __future__ import division
import os
import sys
sys.path.append('../tools')

import tools
import time
test_n = 1000
accurancy = 0
accurancy_full = 0
samples = tools.get_n_samples(tools.test_sameles_path, test_n)
print len(samples.keys())
m=0
n=0
split_accurancy = 0
word_accurancy_no_val = 0
word_accurancy_val = 0
timecost = 0

for name in samples:
	
	print samples.keys().index(name)
	try:
		t = time.time()
		result, full_result, split_index = tools.recognize.recognize(samples[name])
		print 'sample:', name
		print 'direct:', full_result
		print 'result:', result
		timecost += (time.time() - t)
		print 'this sample time cost:%f seconds.' %(time.time() - t)

		split_accurancy += sum([1 if name[i]==result[i] else 0 for i in split_index]) / len(split_index)
		word_accurancy_val += (sum([1 if name[split_index[i]:split_index[i+1]] == result[split_index[i]:split_index[i+1]] else 0 for i in range(len(split_index)-1)]) + (1 if name[split_index[-1]:] == result[split_index[-1]:] else 0)) / len(split_index)
		word_accurancy_no_val += (sum([1 if name[split_index[i]:split_index[i+1]] == full_result[split_index[i]:split_index[i+1]] else 0 for i in range(len(split_index)-1)]) + (1 if name[split_index[-1]:] == full_result[split_index[-1]:] else 0)) / len(split_index)

		if full_result == name:
			m += 1
		if result == name:
			n += 1
		accurancy += tools.compare_result(name, result)
		accurancy_full += tools.compare_result(name, full_result)
	except:
		test_n -= 1
accurancy /= test_n
accurancy_full /= test_n
m/=test_n
n/=test_n
timecost /= test_n
split_accurancy /= test_n
word_accurancy_no_val /= test_n
word_accurancy_val /= test_n

print 'test %d samples, single char accurancy: with validation: %f, without validation: %f\n\t address accurancy: with validation: %f, without validation: %f\n\t word recognize accurancy: with validation %f, without validation %f\n\t address tree layer split accurancy: %f\n\t average single address recognize time cost:%f seconds.' % (test_n, accurancy, accurancy_full, n, m, word_accurancy_val, word_accurancy_no_val, split_accurancy, timecost)
