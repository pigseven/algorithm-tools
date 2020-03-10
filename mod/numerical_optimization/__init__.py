# -*- coding: utf-8 -*-
"""
Created on 2020/3/10 20:53

@Project -> File: algorithm-tools -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

from scipy.optimize import fsolve
from sympy import symbols, diff
from sympy.core.mul import Mul
import numpy as np


# %% 根据自定义一维函数确定二维平面上任一点到该函数的最近点.
def _gen_dy_dx_symbol(func):
	x = symbols('x')
	y = func(x)
	dy_dx_sym = diff(y, x)
	return dy_dx_sym


def _cal_dy_dx(x: float, dy_dx_sym: Mul) -> float:
	return float(dy_dx_sym.subs('x', x))


def cal_closest_point_loc(func, x):
	"""
	计算距离x最近点位置
	"""
	dy_dx_sym = _gen_dy_dx_symbol(func)
	func2solve = lambda p: p - x[0] + _cal_dy_dx(p, dy_dx_sym) * (func(p) - x[1])
	p = fsolve(func2solve, x0 = np.array([x[0]]))
	x_close = [p, func(p)]
	return x_close
	
	
# if __name__ == '__main__':
# 	# %% 自定义函数.
# 	def func(x):
# 		return np.power(x, 2)
	








