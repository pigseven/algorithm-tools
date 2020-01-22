# -*- coding: utf-8 -*-
"""
Created on 2019/12/12 上午10:39

@Project -> File: guodian-desulfuration-optimization -> temporal_serialize.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 将数据在时间上进行连续化处理
"""

from lake.decorator import time_cost
import pandas as pd
import numpy as np
import bisect


def search_nearest_neighbors_in_list(lst, x):
	"""
	寻找x在有序lst中的两侧（或单侧）邻点值.
	:param x: float
	:param lst: list
	:return: neighbors, tuple (left_neighbor, right_neighbor)
	"""
	
	lst_sorted = lst
	lst_sorted.sort(reverse = False)  # 升序排列
	
	try:
		assert lst_sorted == lst
	except:
		raise ValueError('list不是升序排列')
	
	if x in lst:
		return [x]
	else:
		if x <= lst_sorted[0]:
			neighbors = [lst_sorted[0]]
		elif x >= lst_sorted[-1]:
			neighbors = [lst_sorted[-1]]
		else:
			left_idx = bisect.bisect_left(lst_sorted, x) - 1
			right_idx = left_idx + 1
			neighbors = [lst_sorted[left_idx], lst_sorted[right_idx]]
		return neighbors


@time_cost
class DataTemporalSerialization(object):
	"""
	数据时间戳连续化处理, 使用已有记录对未知记录未知进行线性插值填充.
	"""
	
	def __init__(self, data, start_stp, end_stp, stp_step):
		"""
		初始化
		:param data: pd.DataFrame, cols = {'time', vars, ...}, 数据总表
		
		Note:
			1. 待处理的数据表必需含有以"time"字段标记的时间戳;
			2. 数据表中其他字段值必需为int或float类型;
			
		Example:
		------------------------------------------------------------
		# %% 载入数据
		import matplotlib.pyplot as plt
		import sys
		import os
		
		sys.path.append('../..')
		
		from lib import proj_dir
		
		data = pd.read_csv(os.path.join(proj_dir, 'data/provided/weather/raw_records.csv'))
		
		# 查看数据时间戳连续情况
		# 可以发现原始数据时间戳不连续
		plt.plot(list(data['time']))
		
		# %% 时间戳连续化处理
		hr = 3600
		dts = DataTemporalSerialization(data, data['time'].min(), data['time'].max(), hr)
		------------------------------------------------------------
		"""
		
		# 数据格式检查
		try:
			assert 'time' in data.columns
		except Exception:
			raise ValueError('数据表中没有字段"time".')
		
		for col in data.columns:
			dtype = str(data[col].dtypes)
			
			if ('int' not in dtype) & ('float' not in dtype):
				raise ValueError('数据表中字段{}值类型为{}, 不为"int"或"float", 无法进行之后的计算.'.format(col, dtype))
		
		self.expected_stps_list = np.arange(start_stp, end_stp + stp_step, stp_step)
		self.exist_stps_list = data['time'].tolist()
		self.stp_step = stp_step
		self.data = data.copy()
	
	def temporal_serialize(self):
		"""
		时间戳连续化.
		:param device_data: pd.DataFrame, 某id设备数据记录表
		:param stp_list: 连续时间戳list
		:return: device_data: pd.DataFrame, 时间戳连续化后的设备数据记录表
		
		Example:
		------------------------------------------------------------
		data_srlzd = dts.temporal_serialize()
		
		# 可以发现处理后数据时间戳连续
		plt.plot(list(data_srlzd['time']))
		------------------------------------------------------------
		"""
		
		for stp in self.expected_stps_list:
			if stp not in self.exist_stps_list:
				# 从已有数据中提取时间戳两侧最接近的数据
				neighbor_stps = search_nearest_neighbors_in_list(self.exist_stps_list, stp)
				neighbors = self.data[self.data.time.isin(neighbor_stps)]  # 获取前后相邻时间戳的数据
				
				# 根据时间戳距离计算权重并进行线性加权
				if len(neighbor_stps) == 1:
					insert_row = neighbors
				elif len(neighbor_stps) == 2:
					weights = [neighbor_stps[1] - stp, stp - neighbor_stps[0]]
					insert_row = (weights[0] * neighbors.iloc[0, :] + weights[1] * neighbors.iloc[1, :]) / (np.sum(weights))
					insert_row = pd.DataFrame(insert_row).T
				else:
					raise RuntimeError('the length of neighbors is not 1 or 2')
				
				insert_row.loc[0, 'time'] = stp
				self.data = self.data.append(insert_row, ignore_index = True)
		
		# 只选取设定时间戳的数据
		self.data = self.data[self.data.time.isin(self.expected_stps_list)]
		
		# 按照时间戳由小到大顺序排序.
		self.data = self.data.sort_values(by = ['time'], ascending = True)
		
		# 重设index
		self.data.reset_index(drop = True, inplace = True)
		
		return self.data
	
	
