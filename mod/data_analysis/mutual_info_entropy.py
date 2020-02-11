# -*- coding: utf-8 -*-
"""
Created on 2020/2/11 15:14

@Project -> File: algorithm-tools -> mutual_info_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵检测
"""

import numpy as np
import sys

sys.path.append('../..')

from mod.data_binning.data_binning import TwoDimJointBinning

eps = 1e-6


class MutualInfoEntropy(object):
	"""二元变量序列间的互信息熵相关性检测"""
	
	def __init__(self, x, y, value_types):
		"""
		初始化
		:param x: array like 一维数组
		:param y: array like, 一维数组
		:param value_types: dict, like {'x': 'discrete', 'y': 'continuous'}
		"""
		try:
			assert len(x) == len(y)
		except Exception:
			raise ValueError('Series x and y are not in the same length.')
		
		for value_type in value_types.values():
			if value_type not in ['continuous', 'discrete']:
				raise ValueError('Value type {} not in ["continuous", "discrete"].'.format(value_type))
		
		self.x = np.array(x).flatten()
		self.y = np.array(y).flatten()
		self.value_types = value_types
		
	def _data_binning(self, methods, params):
		"""
		数据分箱
		:param methods: dict, like {'x': 'label'; 'y': 'isometric'}
		:param params: dict, like {'x': {}, 'y': {'bins': 100}}
		"""
		samples = np.array([list(self.x), list(self.y)]).T
		tdjb = TwoDimJointBinning(samples, self.value_types)
		allocations, labels, freq_ns = tdjb.marginal_binning(methods, params)
		H = tdjb.joint_binning(allocations, labels, to_array = True)
		
		# 整理结果.
		h_x = freq_ns['x']
		h_y = freq_ns['y']
		
		return h_x, h_y, H
	
	@staticmethod
	def _probability(h):
		freq_sum = np.sum(h)
		probs = h.copy() / freq_sum
		return probs
	
	def _univar_entropy(self, h):
		"""边际分布熵"""
		probs = self._probability(h.copy())
		log_probs = np.log(probs + eps)
		univar_entropy = - np.dot(probs, log_probs)
		return univar_entropy
	
	def _joint_2d_entropy(self, H):
		"""联合分布熵"""
		probs = self._probability(H.copy())
		log_probs = np.log(probs + eps)
		joint_entropy = - np.sum(np.multiply(probs, log_probs))
		return joint_entropy
	
	def mutual_info_entropy(self, methods, params):
		"""
		互信息熵
		
		Example:
		------------------------------------------------------------
		from mod.data_binning import gen_series_samples
		x = gen_series_samples(2000, 'continuous')
		y = gen_series_samples(2000, 'continuous')
		mie = MutualInfoEntropy(x, y, {'x': 'continuous', 'y': 'continuous'})
		
		# 计算互信息熵.
		mutual_info_entro = mie.mutual_info_entropy(
			methods = {'x': 'quasi_chi2', 'y': 'quasi_chi2'},
			params = {'x': {'init_bins': 100, 'final_bins': 50}, 'y': {'init_bins': 100, 'final_bins': 50}}
		)
		------------------------------------------------------------
		"""
		h_x, h_y, H = self._data_binning(methods, params)
		
		# 获取边际熵
		univar_entropy_x = self._univar_entropy(h_x)
		univar_entropy_y = self._univar_entropy(h_y)
		
		# 获取联合分布熵
		joint_entropy = self._joint_2d_entropy(H)
		
		# 互信息熵
		mutual_info_entropy = univar_entropy_x + univar_entropy_y - joint_entropy
		return mutual_info_entropy
