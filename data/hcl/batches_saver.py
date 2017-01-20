# coding: utf-8
# batches_saver.py
# to save the batches frequently used ( 3755 classes, batch size = 1000, size = 32 x 32)
import hcl
import numpy as np

a = hcl.input_data([i for i in range(3755)], direct_info = True)

def batches_saver(ty, base = 0, num_batch = 100, batch_size = 1000):
	for i in range(base, base + num_batch):
		filename = 'hcl/'+ ty + '/bin/'+ str(i)
		x, y = a.next_batch(batch_size, ty)
		x = np.array(x)
		y = np.array(y)
		x.tofile(filename+'.binx')
		y.tofile(filename+'.biny')
		print 'save %d th batch successful!' %i

batches_saver('train', 69)
batches_saver('test', num_batch = 1, batch_size = 10000)