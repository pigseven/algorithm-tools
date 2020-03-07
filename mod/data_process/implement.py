# -*- coding: utf-8 -*-
"""
Created on 2020/2/19 11:26

@Project -> File: pollution-online-data-prediction -> implement.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据填补
"""

import pandas as pd
import numpy as np
import sys

sys.path.append('../..')

from mod.data_process import search_nearest_neighbors_in_list


def data_implement(data: pd.DataFrame, fields2process: list):
	"""
	数据填补
	:param data: pd.DataFrame, 待填补数据表
	:param fields2process: list of strs, 需要进行缺失填补的字段
	
	Example:
	------------------------------------------------------------
	data = data_implement(data, code_types)
	------------------------------------------------------------
	"""
	data = data.copy()
	
	# 逐字段缺失值填补.
	for field in fields2process:
		values = list(data.loc[:, field])
		
		effec_idxs, ineffec_idxs = [], []
		for idx in range(len(values)):
			if np.isnan(values[idx]):
				ineffec_idxs.append(idx)
			else:
				effec_idxs.append(idx)
		
		for idx in ineffec_idxs:
			neighbor_effec_idxs = search_nearest_neighbors_in_list(effec_idxs, idx)
			value2implement = np.mean(data.loc[neighbor_effec_idxs, field])
			data.loc[idx, field] = value2implement
	return data
	
	



