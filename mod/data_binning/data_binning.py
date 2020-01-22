# -*- coding: utf-8 -*-
"""
Created on 2020/1/22 下午3:49

@Project -> File: algorithm-tools -> data_binning.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据分箱
"""

import numpy as np


class UnsupervisedSeriesBinning(object):
	"""一维无监督序列分箱"""
	
	def __init__(self, x):
		"""初始化"""
		self.x = np.array(x).flatten()
	
	@staticmethod
	def zero_one_normalize(x):
		x_min, x_max = np.min(x), np.max(x)
		x = (x - x_min) / (x_max - x_min)
		return x
	
	def _get_stat_params(self):
		"""计算1维数据x的统计学参数"""
		# 计算均值方差.
		mean = np.mean(self.x)
		std = np.std(self.x)
		
		# 计算四分位数.
		q1, q2, q3 = np.percentile(self.x, (25, 50, 75), interpolation = 'midpoint')
		iqr = abs(q3 - q1)
		
		# 汇总结果.
		stat_params = {
			'mean': mean,
			'std': std,
			'percentiles': {
				'q1': q1,  # 下25%位
				'q2': q2,  # 中位数
				'q3': q3,  # 上25%位
				'iqr': iqr
			}
		}
		
		return stat_params
	
	@property
	def stat_params(self):
		return self._get_stat_params()
	
	def isometric_binning(self, bins):
		"""
		等距分箱，适用于对类似高斯型数据进行分箱
		:param bins: int > 0, 分箱个数
		:return:
			freq_ns: list of ints, 箱子中的频数
			densities: list of floats, 密度
			labels: list of strs or ints, 各箱的标签
		"""
		
		# 计算数据整体的上下四分位点以及iqr距离.
		# 按照上下四分位点外扩1.5倍iqr距离获得分箱外围边界以及每个箱子长度
		percentiles = self.stat_params['percentiles']
		q3, q1, iqr = percentiles['q3'], percentiles['q1'], percentiles['iqr']
		binning_range = [
			max(np.min(self.x), q1 - 1.5 * iqr),
			min(np.max(self.x), q3 + 1.5 * iqr)
		]
		
		# 分箱.
		freq_ns, intervals = np.histogram(self.x, bins, range = binning_range)
		labels = intervals[1:]  # **以每个分箱区间右侧为label
		
		# 转为list类型.
		freq_ns = list(freq_ns)
		labels = list(labels)
		
		return freq_ns, labels
	
	def quasi_chi2_binning(self, init_bins, final_bins, merge_freq_thres = None):
		"""
		拟卡方分箱(无监督), 适用于对存在多个分布峰的非高斯类型数据进行分箱
		:param init_bins: int > 0, 初始分箱个数
		:param final_bins: int > 0, 最终分箱个数
		:param merge_freq_thres: float > 0.0, 用于判断是否合并的箱频率密度阈值, 一般设为(样本量 / init_bins / 10)
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs, 各箱的标签
		"""
		
		# 设置默认合并频率阈值.
		if merge_freq_thres is None:
			merge_freq_thres = len(self.x) / init_bins / 10
		
		# 首先进行粗分箱.
		init_freq_ns, init_labels = self.isometric_binning(init_bins)
		init_densities = init_freq_ns  # 这里使用箱频率密度表示概率分布意义上的密度
		init_box_lens = [1] * init_bins
		
		# 根据相邻箱密度差异判断是否合并箱.
		bins = init_bins
		freq_ns = init_freq_ns
		labels = init_labels
		box_lens = init_box_lens
		densities = init_densities
		while True:
			do_merge = 0
			# 在一次循环中优先合并具有最高相似度的箱.
			similarities = {}
			for i in range(bins - 1):
				j = i + 1
				density_i, density_j = densities[i], densities[j]
				s = abs(density_i - density_j)  # 密度相似度
				
				if s <= merge_freq_thres:
					similarities[i] = s
					do_merge = 1
				else:
					continue
			
			if (do_merge == 0) | (bins == final_bins):
				break
			else:
				similarities = sorted(similarities.items(), key = lambda x: x[1], reverse = False)  # 升序排列
				i = list(similarities[0])[0]
				j = i + 1
				
				# 执行i和j箱合并, j合并到i箱
				freq_ns[i] += freq_ns[j]
				box_lens[i] += box_lens[j]
				densities[i] = freq_ns[i] / box_lens[i]  # 使用i、j箱混合后的密度
				labels[i] = labels[j]
				
				freq_ns = freq_ns[: j] + freq_ns[j + 1:]
				densities = densities[: j] + densities[j + 1:]
				labels = labels[: j] + labels[j + 1:]
				box_lens = box_lens[: j] + box_lens[j + 1:]
				
				bins -= 1
		
		return freq_ns, labels


