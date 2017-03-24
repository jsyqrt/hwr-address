# coding: utf-8
# test_recognize.py to test `../src/recognize.py`
from __future__ import division
import os
import sys
sys.path.append('../tools')

import tools

test_n = 1000
accurancy = 0
accurancy_full = 0
samples = tools.get_n_samples(tools.test_sameles_path, test_n)
print len(samples.keys())
m=0
n=0
for name in samples:
	print samples.keys().index(name)
	result, full_result = tools.recognize.recognize(samples[name])
	print 'sample:', name
	print 'direct:', full_result
	print 'result:', result
	if full_result == name:
		m += 1
	if result == name:
		n += 1
	accurancy += tools.compare_result(name, result)
	accurancy_full += tools.compare_result(name, full_result)

accurancy /= test_n
accurancy_full /= test_n
m/=test_n
n/=test_n
print 'test %d samples, char accurancy: validated: %f, direct: %f\n address accurancy: validated: %f, direct: %f' % (test_n, accurancy, accurancy_full, n, m)
