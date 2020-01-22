# -*- coding: utf-8 -*-
"""
Created on 2020/1/15 下午3:13

@Project -> File: algorithm-tools -> mutual_info_entropy_test.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 互信息熵测试
"""

from lake.decorator import time_cost
import numpy as np
import copy
import sys

sys.path.append('../..')

from mod.data_binning.data_binning import UnsupervisedSeriesBinning

eps = 1e-6


class MutualInfoEntropy(object):
	"""二元序列变量间的互信息熵相关性检测"""
	
	def __init__(self, x, y, method, **kwargs):
		self.x = np.array(x).flatten()
		self.y = np.array(y).flatten()
		
		try:
			assert len(self.x) == len(self.y)
		except Exception:
			raise ValueError('Series x and y are not in the same length.')
		
		self._series_binning(method = method, **kwargs)
	
	def _series_binning(self, method = 'quasi_chi2', **kwargs):
		"""
		一维序列各自分箱, 获得两个变量各自分箱的频率计数和标签.
		:param method: str from {'quasi_chi2', 'isometric'}, 选用的分箱方法
		:param kwargs:
			if method == 'quasi_chi2':
				init_bins: int > 0, 初始分箱个数
				final_bins: init > 0, 最终分箱个数下限
			elif method == 'isometric':
				bins: int > 0, 分箱个数
		"""
		
		usb_x = UnsupervisedSeriesBinning(self.x)
		usb_y = UnsupervisedSeriesBinning(self.y)
		
		if method == 'quasi_chi2':
			self.x_freq_ns, self.x_labels = usb_x.quasi_chi2_binning(**kwargs)
			self.y_freq_ns, self.y_labels = usb_y.quasi_chi2_binning(**kwargs)
		elif method == 'isometric':
			self.x_freq_ns, self.x_labels = usb_x.isometric_binning(**kwargs)
			self.y_freq_ns, self.y_labels = usb_y.isometric_binning(**kwargs)
		else:
			raise ValueError('Unknown method {}.'.format(method))
	
	def joint_binning(self, x, y):
		"""计算联合分布分箱"""
		H, _, _ = np.histogram2d(
			x, y, bins = [[np.min(x)] + self.x_labels, [np.min(y)] + self.y_labels]
		)
		return H
	
	@staticmethod
	def _probability(freq_ns):
		freq_sum = np.sum(freq_ns)
		probs = freq_ns.copy() / freq_sum
		return probs
	
	def _univar_entropy(self, freq_ns):
		"""边际分布熵"""
		probs = self._probability(freq_ns.copy())
		log_probs = np.log(probs + eps)
		univar_entropy = - np.dot(probs, log_probs)
		return univar_entropy
	
	def _joint_2d_entropy(self, H):
		"""联合分布熵"""
		probs = self._probability(H.copy())
		log_probs = np.log(probs + eps)
		joint_entropy = - np.sum(np.multiply(probs, log_probs))
		return joint_entropy
	
	def mutual_info_entropy(self, x, y):
		"""
		互信息熵
		
		Example:
		————————————————————————————————————————————————————————————
		# %% 生成随机时间序列数据样本.
		x = np.hstack((np.random.normal(0, 1.0, 5000), np.random.normal(20, 1.0, 5000)))
		y = np.hstack((np.random.normal(2.0, 1.0, 5000), np.random.normal(10, 1.0, 5000)))
	
		# %% 互信息熵计算.
		miet = MutualInfoEntropy(x, y, method = 'quasi_chi2', init_bins = 120, final_bins = 50)
		mutual_info_entropy = miet.mutual_info_entropy(x, y)
		————————————————————————————————————————————————————————————
		"""
		
		# 获取边际熵.
		univar_entropy_x = self._univar_entropy(self.x_freq_ns)
		univar_entropy_y = self._univar_entropy(self.y_freq_ns)
		
		# 获取联合分布熵.
		H = self.joint_binning(x, y)
		joint_entropy = self._joint_2d_entropy(H)
		
		# 互信息熵.
		mutual_info_entropy = univar_entropy_x + univar_entropy_y - joint_entropy
		return mutual_info_entropy
	
	@time_cost
	def time_delayed_mutual_info_entropy(self, lags):
		"""
		带有时间延迟的互信息熵检测.
		:param lags: list, 延迟时间步序列, [-max_lag, -max_lag + lag_step, ..., max_lag]
		:return: td_corr_dict, dict, 记录各延迟时间步上的互信息熵检测结果
		
		Example:
		————————————————————————————————————————————————————————————
		lag_step = 1
		max_lag = 1000
		lags = np.arange(-max_lag, max_lag + lag_step, lag_step).tolist()
		td_mie_dict = miet.time_delayed_mutual_info_entropy(lags)
		
		# %% 画出时间延迟互信息熵检测结果.
		import matplotlib.pyplot as plt
		bg_split = 800
		
		# 计算背景均值和标准差, 使用3-sigma准则判断是否明显相关.
		bg_values = [abs(v) for k, v in td_mie_dict.items() if abs(k) > bg_split]  # ** 注意取了绝对值
		mean, std = np.mean(bg_values), np.std(bg_values)
		sig_thres = mean + 3 * std
		
		# 画图.
		plt.plot(list(td_mie_dict.keys()), np.abs(list(td_mie_dict.values())))
		plt.fill_between(list(td_mie_dict.keys()), np.abs(list(td_mie_dict.values())))
		plt.plot([-max_lag, max_lag], [0, 0], 'k--', linewidth = 0.3)
		plt.plot([-max_lag, max_lag], [sig_thres, sig_thres], 'r', linewidth = 0.5)
		plt.xlim([-max_lag, max_lag])
		
		max_ie = np.max(list(td_mie_dict.values()))
		min_ie = np.min(list(td_mie_dict.values()))
		plt.ylim([max(min_ie, 0.0), max_ie])
		
		plt.ticklabel_format(useOffset = False)
		plt.xticks(fontsize = 6)
		plt.yticks(fontsize = 6)
		plt.tight_layout()
		plt.subplots_adjust(top = 0.933)
		————————————————————————————————————————————————————————————
		"""
		
		data_len = len(self.x)
		td_mie_dict = {}
		for lag in lags:
			lag_remain = np.abs(lag) % data_len  # 整除后的余数
			
			if lag_remain == 0:
				x_td = copy.deepcopy(self.x)
				y_td = copy.deepcopy(self.y)
			else:
				if lag > 0:
					x_td = copy.deepcopy(self.x)
					y_td = np.hstack((self.y[lag_remain:], self.y[:lag_remain]))
				else:
					x_td = np.hstack((self.x[lag_remain:], self.x[:lag_remain]))
					y_td = copy.deepcopy(self.y)
			
			mutual_info_entropy = self.mutual_info_entropy(x_td, y_td)
			td_mie_dict[lag] = mutual_info_entropy
		return td_mie_dict
	
	

