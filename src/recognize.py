# coding: utf-8
# recognize.py
# to combine all functions defined before to realize the recogniztion of HCCR.

from preprocess import preprocess

from tfk import keyword_recognizer
from tf3755 import full_recognizer
import at

def __init__():
	pass

k = keyword_recognizer()
f = full_recognizer()

def recognize(img):
	charlist = preprocess(img)
	k_char_list = k.recognize(charlist)
	full_char_list = f.recognize(charlist)
	result = at.validate(k_char_list, full_char_list)
	return result