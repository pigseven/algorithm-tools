# -*- coding: utf-8 -*-
"""
Created on 2020/3/24 15:29

@Project -> File: algorithm-tools -> step_0_process_raw.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 原始数据处理
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys, os

sys.path.append('../..')

from lib import proj_dir
from mod.data_process.temporal_serialize import DataTemporalSerialization
from mod.data_process.implement import data_implement
from mod.data_process.normalize_and_denoise import normalize_cols, denoise_cols
from mod.data_process.replace_singulars import replace_singulars


if __name__ == '__main__':
	# 该脚本用于模拟使用mod.data_process中各数据处理功能函数.
	
	# %% 载入数据.
	from lib.data_process.tmp import data_raw
	
	# %% 数据时间戳连续化处理.
	start_stp, end_stp = 1462413600, 1557028800
	stp_step = 3600
	cols2serialze = ['pm25', 'pm10', 'so2', 'co', 'no2', 'o3', 'aqi',
	                 'weather', 'ws', 'wd', 'temp', 'sd', 'month', 'weekday', 'clock_num']
	dts = DataTemporalSerialization(data_raw, start_stp, end_stp, stp_step, cols2serialze)
	data, miss_n = dts.temporal_serialize(categorical_cols = ['weather', 'wd', 'ws', 'month', 'weekday', 'clock_num'])
	
	# %% 数据缺失值填补.
	data = data_implement(data)
	
	# 保存数据.
	data.to_csv(os.path.join(proj_dir, 'data/runtime/data_srlzd.csv'), index = False)
	
	# %% 数据归一化、去噪和异常值处理.
	cols_bounds = {
		'pm25': [0, 600],
		'pm10': [0, 1400],
		'so2': [0, 600],
		'co': [0, 8],
		'no2': [0, 250],
		'o3': [0, 400],
		'aqi': [0, 1300],
		'weather': [0, 19],
		'ws': [0, 10],
		'wd': [0, 18],
		'temp': [-40, 60],
		'sd': [0, 110],
		'month': [0, 12],
		'weekday': [0, 7],
		'clock_num': [0, 24]}
	data = normalize_cols(data, cols_bounds)
	
	cols2denoise = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co', 'aqi', 'ws', 'temp', 'sd']
	window_size, order = 3, 1
	data = denoise_cols(data, cols2denoise, window_size, order)
	
	cols2check = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co', 'aqi']
	singular_value = 0.0
	data = replace_singulars(data, cols2check, singular_value)
	
	# 保存数据.
	data.to_csv(os.path.join(proj_dir, 'data/runtime/data_denoised.csv'), index = False)
	

