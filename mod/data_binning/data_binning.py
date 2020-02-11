# -*- coding: utf-8 -*-
"""
Created on 2020/2/11 14:34

@Project -> File: algorithm-tools -> one_dim_binning.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据分箱
"""

import numpy as np
import collections
import warnings
import sys

sys.path.append('../..')

from mod.tool.operation import sort_dict_by_keys


class OneDimSeriesBinning(object):
	"""一维序列分箱"""
	
	def __init__(self, x, x_type):
		"""
		初始化
		:param x: list or array like, 待分箱序列
		:param x_type: str in ['continuous', 'discrete']
		"""
		if x_type not in ['continuous', 'discrete']:
			raise ValueError('The param x_type {} not in ["continuous", "discrete"].'.format(x_type))
		
		self.x = np.array(x).flatten()  # flatten处理
		self.x_type = x_type
	
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
	
	def _isometric_binning(self, bins):
		"""
		等距分箱，适用于对类似高斯型数据进行分箱
		:param bins: int > 0, 分箱个数
		:return:
			freq_ns: list of ints, 箱子中的频数
			densities: list of floats, 密度
			labels: list of strs or ints, 各箱的标签
		"""
		if self.x_type == 'discrete':
			warnings.warn('Invalid x_type: "discrete", self.isometric_binning is better for x_type = "continuous", '
						  'please switch to self.label_binning.')
		
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
	
	def _quasi_chi2_binning(self, init_bins, final_bins, merge_freq_thres = None):
		"""
		拟卡方分箱(无监督), 适用于对存在多个分布峰的非高斯类型连续变量数据进行分箱
		:param init_bins: int > 0, 初始分箱个数
		:param final_bins: int > 0, 最终分箱个数
		:param merge_freq_thres: float > 0.0, 用于判断是否合并的箱频率密度阈值, 一般设为(样本量 / init_bins / 10)
		:return:
			freq_ns: list of ints, 箱子中的频数
			labels: list of strs, 各箱的标签
		"""
		if self.x_type == 'discrete':
			warnings.warn('Invalid x_type: "discrete", self.quasi_chi2_binning is better for x_type = "continuous", '
						  'please switch to self.label_binning.')
		
		# 设置默认合并频率阈值.
		if merge_freq_thres is None:
			merge_freq_thres = len(self.x) / init_bins / 10
		
		# 首先进行粗分箱.
		init_freq_ns, init_labels = self._isometric_binning(init_bins)
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
	
	def _label_binning(self):
		"""
		根据离散数据标签进行分箱
		"""
		if self.x_type == 'continuous':
			warnings.warn('Invalid x_type: "continuous", self.label_binning is better for x_type = "discrete", '
						  'please switch to self.label_binning or self.quasi_chi2_binning.')
		
		labels = sorted(list(set(list(self.x))))  # **按照值从小到大排序
		freq_ns = list(np.zeros(len(labels)))
		for i in range(len(self.x)):
			label = self.x[i]
			freq_ns[labels.index(label)] += 1
		return freq_ns, labels
	
	def series_binning(self, method, params = None):
		"""
		序列分箱
		:param method:
		:param params:
		
		Example:
		------------------------------------------------------------
		from mod.data_binning import gen_series_samples
	
		# 离散值分箱测试.
		x_0 = gen_series_samples(sample_len = 2000, value_type = 'discrete')
		odsb_0 = OneDimSeriesBinning(x_0, x_type = 'discrete')
		freq_ns_0, labels_0 = odsb_0.series_binning(method = 'label')
		allocate_records_0 = odsb_0.allocate_samples(labels_0)
	
		# 连续值分箱测试.
		x_1 = gen_series_samples(sample_len = 2000, value_type = 'continuous')
		odsb_1 = OneDimSeriesBinning(x_1, x_type = 'continuous')
		freq_ns_1, labels_1 = odsb_1.series_binning(
			method = 'quasi_chi2', params = {'init_bins': 100, 'final_bins': 20}
		)
		allocate_records_1 = odsb_1.allocate_samples(labels = labels_1)
		------------------------------------------------------------
		"""
		if method == 'isometric':
			freq_ns, labels = self._isometric_binning(params['bins'])
		elif method == 'quasi_chi2':
			freq_ns, labels = self._quasi_chi2_binning(params['init_bins'], params['final_bins'])
		elif method == 'label':
			freq_ns, labels = self._label_binning()
		else:
			raise ValueError('Invalid method {}.'.format(method))
		return freq_ns, labels
	
	def allocate_samples(self, labels = None):
		"""
		获取样本中每个元素对应的分箱label
		"""
		allocate_records = collections.defaultdict(list)  # 记录每个lable下面的样本index
		if self.x_type == 'discrete':  # 离散值分箱结果收集
			for i in range(len(self.x)):
				label = self.x[i]
				allocate_records[label].append(i)
		elif self.x_type == 'continuous':  # 连续值分箱结果收集
			# 连续类型值必须有lables.
			if labels is None:
				raise ValueError('Labels must be provided while self.x_type == "continuous".')
			
			for i in range(len(self.x)):
				value = self.x[i]
				
				# 确定连续值分箱的label.
				is_label_found = 0
				for j in range(len(labels) - 1):
					label_left = labels[j]
					label_right = labels[j + 1]
					if (label_left >= value) & (j == 0):
						allocate_records[label_left].append(i)
						is_label_found = 1
						break
					elif label_left < value <= label_right:
						allocate_records[label_right].append(i)
						is_label_found = 1
						break
					elif (label_right < value) & (j == len(labels) - 2):
						allocate_records[label_right].append(i)
						is_label_found = 1
						break
				if is_label_found == 0:
					raise RuntimeError('Label not found for i = {}, value = {}.'.format(i, value))
		
		allocate_records = sort_dict_by_keys(allocate_records)
		
		return allocate_records


class TwoDimJointBinning(object):
	"""二维数据联合分箱"""
	
	def __init__(self, samples, value_types):
		"""
		初始化
		:param samples: array like, shape = (-1, 2), 待分箱序列
		:param value_types: dict, like {'x': 'continuous', 'y': 'discrete'}
		"""
		for x_type in value_types.values():
			if x_type not in ['continuous', 'discrete']:
				raise ValueError('x_type {} not in ["continuous", "discrete"].'.format(x_type))
		
		if samples.shape[1] != 2:
			raise ValueError('Samples must be 2d array like.')
		
		self.samples = samples
		self.value_types = value_types
	
	@staticmethod
	def _check_method_value_type(method, value_type):
		"""检查分箱方法与值类型是否匹配"""
		if value_type == 'discrete':
			if method not in ['label']:
				raise ValueError('Binning method {} is not in ["label"] while value_type is {}.'.format(method, value_type))
		elif value_type == 'continuous':
			if method not in ['isometric', 'quasi_chi2']:
				raise ValueError(
					'Binning method {} is not in ["isometric", "quasi_chi2"] while value_type is {}.'.format(method, value_type))
	
	@staticmethod
	def _fill_empty_keys(adict, keys4fill):
		for k in keys4fill:
			if k not in adict.keys():
				adict.update({k: []})
		return adict
	
	def marginal_binning(self, methods, params):
		"""
		针对x和y序列边际分布进行分箱
		:param methods: dict, like {'x': 'method_x', 'y': 'method_y'}
		:param params: dict, like {'x': {'param_0': param_0}, 'y': {'param_1': param_1, 'param_2': param_2}}
		:return:
		"""
		# x和y方向分箱.
		vars = ['x', 'y']
		allocations = collections.defaultdict(dict)
		labels = collections.defaultdict(list)
		freq_ns = collections.defaultdict(list)
		for var in vars:
			method = methods[var]
			value_type = self.value_types[var]
			self._check_method_value_type(method, value_type)
			odsb = OneDimSeriesBinning(self.samples[:, vars.index(var)], x_type = value_type)
			freqs, labs = odsb.series_binning(method, params[var])
			allocs = odsb.allocate_samples(labels = labs)
			
			# 对空箱进行填补.
			allocations[var] = self._fill_empty_keys(allocs, labs)
			labels[var] = labs
			freq_ns[var] = freqs
		
		return allocations, labels, freq_ns
		
	def joint_binning(self, allocations, labels, to_array = False):
		"""
		对每列序列进行分箱
		# :param methods: dict, each value must be in {'isometric', 'quasi_chi2', 'label'}
		# :param x_params: dict, 分箱method函数参数设置
		# :param y_params: dict, 同x_params
		
		Example:
		------------------------------------------------------------
		from mod.data_binning import gen_two_dim_samples

		samples = gen_two_dim_samples(samples_len = 2000, value_types = ['continuous', 'continuous'])
		tdjb = TwoDimJointBinning(samples, value_types = {'x': 'continuous', 'y': 'continuous'})
		
		methods = {'x': 'isometric', 'y': 'quasi_chi2'}
		params = {'x': {'bins': 100}, 'y': {'init_bins': 100, 'final_bins': 30}}
		allocations, labels, freq_ns = tdjb.marginal_binning(methods, params)
		joint_binning_results = tdjb.joint_binning(allocations, labels, to_array = True)
		------------------------------------------------------------
		"""
		allocs_x, allocs_y = allocations['x'], allocations['y']
		labels_x, labels_y = labels['x'], labels['y']

		# 联合分箱.
		joint_binning_results = {}
		for label_x in allocs_x.keys():
			if allocs_x[label_x] == []:
				allocs_y = {}
			else:
				idxs = allocs_x[label_x]
				odsb_y = OneDimSeriesBinning(self.samples[idxs, 1], x_type = self.value_types['y'])
				allocs_y = odsb_y.allocate_samples(labels = labels_y)
			# 对空箱进行填补.
			allocs_y = self._fill_empty_keys(allocs_y, labels_y)

			# 计算箱子对应频数.
			for label_y in allocs_y.keys():
				allocs_y[label_y] = len(allocs_y[label_y])
			joint_binning_results[label_x] = sort_dict_by_keys(allocs_y)
		joint_binning_results = sort_dict_by_keys(joint_binning_results)

		if to_array:
			joint_binning_results = np.array([list(joint_binning_results[k].values()) for k in joint_binning_results.keys()])

		return joint_binning_results
