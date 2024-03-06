import os
import ctypes


def gcc_complie(c_path):
	so_path = c_path[:-2]+'.so'
	os.system('gcc -o ' + so_path + ' -shared -fPIC ' + c_path + ' -O2')
	return ctypes.cdll.LoadLibrary(so_path)
