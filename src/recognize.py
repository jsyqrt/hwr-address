# coding: utf-8
# recognize.py
# to combine all functions defined before to realize the recogniztion of HCCR.

import input_data
from hccrn import recognizer

k = recognizer(*input_data.arguments['kwd'])
f = recognizer(*input_data.arguments['full'])
t = input_data.at.at()

def recognize(img_file_path):
	charlist = input_data.preprocess(img_file_path)
	k_char_list = k.recognize(charlist, input_data.keyword_list)
	full_char_list = f.recognize(charlist, input_data.full_list)
	#result = input_data.at.validate(t, k_char_list, full_char_list)
	result = input_data.at.validate(t, k_char_list, full_char_list, charlist)
	return result

#print recognize('../data/sample/test/安徽省安庆市望江县杨湾镇曾墩村委会.jpg')
