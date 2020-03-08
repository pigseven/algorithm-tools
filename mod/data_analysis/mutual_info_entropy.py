# -*- coding: utf-8 -*-
"""
Created on 2020/3/8 15:30

@Project -> File: algorithm-tools -> mutual_info_entropy.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵检测
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import numpy as np
import copy
import sys

sys.path.append('../..')

from mod.data_binning import value_types_available, methods_available
from mod.data_binning.series_binning import SeriesBinning
from mod.data_binning.joint_binning import JointBinning

eps = 1e-6


class MutualInfoEntropy(object):
	"""二元变量序列间的互信息熵相关性检测"""
	
	def __init__(self, x: np.array, y: np.array, value_types: list):
		"""
		初始化
		:param x: array like 一维数组
		:param y: array like, 一维数组
		:param value_types: list, x和y的值类型, like ['discrete', 'continuous']
		"""
		if len(x) != len(y):
			raise ValueError('Series x and y are not in the same length')
		
		for value_type in value_types:
			if value_type not in value_types_available:
				raise ValueError('Value type {} is not in value_types_available = {}'.format(value_type, value_types_available))
		
		self.D = 2
		self.x = np.array(x).flatten()
		self.y = np.array(y).flatten()
		self.N = len(self.x)
		self.value_types = value_types
	
	@staticmethod
	def _probability(freq_ns):
		freq_sum = np.sum(freq_ns)
		probs = freq_ns.copy() / freq_sum
		return probs
	
	def _univar_entropy(self, freq_ns):
		"""边际分布熵"""
		
		eps = 1e-6
		probs = self._probability(freq_ns.copy())
		log_probs = np.log(probs + eps)
		univar_entropy = - np.dot(probs, log_probs)
		
		return univar_entropy
	
	def _joint_2d_entropy(self, H):
		"""联合分布熵"""
		
		eps = 1e-6
		probs = self._probability(H.copy())
		log_probs = np.log(probs + eps)
		joint_entropy = - np.sum(np.multiply(probs, log_probs))
		
		return joint_entropy
	
	def _check_value_type_and_method(self, methods):
		for i in range(self.D):
			value_type = self.value_types[i]
			method = methods[i]
			try:
				assert method in methods_available[value_type]
			except:
				raise ValueError('ERROR: method {} does not match value_type {}'.format(method, value_type))
	
	@time_cost
	def cal_mutual_info_entropy(self, methods: list, params: list):
		"""计算互信息熵"""
		self._check_value_type_and_method(methods)
		
		# 各维度边际熵.
		series_binning_x_ = SeriesBinning(self.x, x_type = self.value_types[0])
		freq_ns_x_, _ = series_binning_x_.series_binning(method = methods[0], params = params[0])
		series_binning_y_ = SeriesBinning(self.y, x_type = self.value_types[1])
		freq_ns_y_, _ = series_binning_y_.series_binning(method = methods[1], params = params[1])
		
		univar_entropy_x_ = self._univar_entropy(freq_ns_x_)
		univar_entropy_y_ = self._univar_entropy(freq_ns_y_)
		
		# 联合分布熵.
		data_ = np.vstack((self.x, self.y)).T
		joint_binning_ = JointBinning(data_, self.value_types, methods = methods, params = params)
		hist, _ = joint_binning_.joint_binning()
		joint_entropy_ = self._joint_2d_entropy(hist)
		
		# 互信息熵
		mutual_info_entropy = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
		
		return mutual_info_entropy
	
	@time_cost
	def cal_time_delayed_mutual_info_entropy(self, methods: list, params: list, lags: list):
		"""含有时滞的互信息熵"""
		self._check_value_type_and_method(methods)
		
		# 各维度边际熵.
		series_binning_x_ = SeriesBinning(self.x, x_type = self.value_types[0])
		freq_ns_x_, edges_x_ = series_binning_x_.series_binning(method = methods[0], params = params[0])
		series_binning_y_ = SeriesBinning(self.y, x_type = self.value_types[1])
		freq_ns_y_, edges_y_ = series_binning_y_.series_binning(method = methods[1], params = params[1])
		
		univar_entropy_x_ = self._univar_entropy(freq_ns_x_)
		univar_entropy_y_ = self._univar_entropy(freq_ns_y_)
		
		edges_ = [edges_x_, edges_y_]
		# edges_len_ = [len(edges_x_), len(edges_y_)]
		
		# 计算时滞联合分布熵.
		td_corr_dict = {}
		for lag in lags:
			# 序列平移.
			lag_remain = np.abs(lag) % len(self.x)  # 整除后的余数
			x_td = copy.deepcopy(self.x)
			y_td = copy.deepcopy(self.y)
			
			if lag_remain == 0:
				pass
			else:
				if lag > 0:
					y_td = np.hstack((y_td[lag_remain:], y_td[:lag_remain]))
				else:
					x_td = np.hstack((x_td[lag_remain:], x_td[:lag_remain]))
			
			data_ = np.vstack((x_td, y_td)).T
			
			# 在各个维度上将数据值向label进行插入, 返回插入位置.
			insert_locs_ = np.zeros_like(data_, dtype = int)
			for d in range(self.D):
				insert_locs_[:, d] = np.searchsorted(edges_[d], data_[:, d], side = 'left')
			
			# 将高维坐标映射到一维坐标上, 然后统计各一维坐标上的频率.
			edges_len_ = list(np.max(insert_locs_, axis = 0) + 1)
			ravel_locs_ = np.ravel_multi_index(insert_locs_.T, dims = edges_len_)
			hist_ = np.bincount(ravel_locs_, minlength = np.array(edges_len_).prod())
			
			# reshape转换形状.
			hist_ = hist_.reshape(edges_len_)
			
			# 计算联合分布熵和互信息熵.
			joint_entropy_ = self._joint_2d_entropy(hist_)
			mutual_info_entropy_ = univar_entropy_x_ + univar_entropy_y_ - joint_entropy_
			td_corr_dict[lag] = mutual_info_entropy_
		
		return td_corr_dict
