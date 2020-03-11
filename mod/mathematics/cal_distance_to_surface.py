# -*- coding: utf-8 -*-
"""
Created on 2020/3/11 15:16

@Project -> File: algorithm-tools -> cal_distance_to_surface.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 高维空间中点离曲面的距离计算
"""

from scipy.optimize import fsolve
from sympy import symbols, diff
import numpy as np
import sympy

eps = 1e-12
inf = 1e12


class CalPartialDerivs(object):
	"""
	计算标准型函数的偏导数
	* func必须为 f(x0, x1, ..., xn) = 0的形式
	* func的维数为f_dim, 且f_dim = n + 1
	"""
	
	def __init__(self, func, f_dim: int):
		self.func = func
		self.f_dim = f_dim
		
		# 生成变量x、y以及函数g和偏导数符号.
		self.x_symbols = self._gen_x_symbols()
		self.y_symbol = self.func(self.x_symbols)
		self.g_symbol = sympy.solve(self.y_symbol, self.x_symbols[-1])[0]
		self.pd_symbols = []
		for i in range(self.f_dim):
			self.pd_symbols.append(diff(self.y_symbol, self.x_symbols[i]))
		
		# 提取函数g和各偏导数中所用到的变量x.
		self.g_vars = self._extract_x_symbols(self.g_symbol)
		self.pd_vars = []
		for i in range(self.f_dim):
			pd_func_ = self.pd_symbols[i]  # 获取该维度变量xi对应的偏导数符号式p_f / p_xi
			vars_ = []
			for j in range(self.f_dim):
				if symbols('x_{}'.format(j)) in pd_func_.free_symbols:
					vars_.append('x_{}'.format(j))  # 记录下p_f / p_xi中出现的各变量x
			self.pd_vars.append(vars_)
		
	def _gen_x_symbols(self):
		x_symbols = []
		for i in range(self.f_dim):
			x_symbols.append(symbols('x_{}'.format(i)))
		return x_symbols
		
	def _extract_x_symbols(self, expression) -> list:
		"""
		提取表达式expression中的所有x变量符号名
		"""
		vars = []
		for j in range(self.f_dim):
			if symbols('x_{}'.format(j)) in expression.free_symbols:
				vars.append('x_{}'.format(j))  # 记录下p_f / p_xi中出现的各变量x
		return vars
	
	def cal_pd_values(self, x: list) -> (list, list):
		"""
		计算func对x各维度上的偏导数值, 注意func已经是func(x) = 0的标准形式
		"""
		assert len(x) == self.f_dim - 1
		
		subs_dict = {}
		for var in self.g_vars:
			subs_dict[var] = x[int(var.split('_')[1])]
		x_end = float(self.g_symbol.subs(subs_dict))
		x.append(x_end)
		
		pd_values = []
		for i in range(self.f_dim):
			subs_dict_ = {}
			for var in self.pd_vars[i]:
				subs_dict_[var] = x[int(var.split('_')[1])]
			pd_ = float(self.pd_symbols[i].subs(subs_dict_))
			pd_values.append(pd_)
		
		return x, pd_values
	
	
def cal_distance2surface(func, f_dim: int, xps: list, x0: list) -> np.array:
	"""
	计算高维空间中点xps离函数func(x) = 0构成的曲面的距离和最近点坐标
	"""
	cpd = CalPartialDerivs(func, f_dim)
	
	def _eqs2solve(x):
		x = list(x).copy()
		x, pd_values = cpd.cal_pd_values(x)
		eqs = []
		for i in range(f_dim - 1):
			if pd_values[-1] == 0:
				pd_values[-1] = eps
			if pd_values[i] == 0:
				pd_values[i] = eps
			e_ = (xps[i] - x[i]) / (pd_values[i] / pd_values[-1]) - xps[-1] + x[-1]
			eqs.append(e_)
		return eqs
	
	root = fsolve(_eqs2solve, np.array(x0))
	
	# 计算曲线上最近邻点.
	x_opt, _ = cpd.cal_pd_values(list(root))
	dist = np.linalg.norm(x_opt - np.array(xps))
	
	return x_opt, dist
		

if __name__ == '__main__':
	# %% 定义函数和参数.
	def func(x: list):
		y = x[1] - 2 * x[0]
		return y
	
	# %% 测试类.
	f_dim = 2
	xps = [0.5, 3]
	x0 = [1.5]
	x_opt, dist = cal_distance2surface(func, f_dim, xps, x0)
	
	# %% 画图.
	import matplotlib.pyplot as plt
	x = np.arange(-3, 3 + 0.1, 0.1)
	y = 2 * x
	plt.figure(figsize = [8, 8])
	plt.plot(x, y)
	plt.scatter(xps[0], xps[1], s = 12)
	plt.scatter(x_opt[0], x_opt[1], s = 12 , c = 'black')
	plt.xlim([-3, 3])
	plt.ylim([-3, 3])
	plt.grid(True)
	
	




