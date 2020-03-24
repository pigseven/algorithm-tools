# -*- coding: utf-8 -*-
"""
Created on 2020/3/24 16:05

@Project -> File: algorithm-tools -> step_1_data_analysis.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据分析
"""

import logging

logging.basicConfig(level = logging.INFO)

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys, os

sys.path.append('../..')

from lib import proj_dir


if __name__ == '__main__':
	# %% 载入数据.
	from lib.data_process.tmp import data_denoised as data
	
	# %% pairplot作图.
	cols = [
		'pm10', 'pm25', 'o3', 'so2', 'co', 'no2', 'aqi',
		'clock_num', 'weekday', 'month', 'sd', 'weather', 'temp', 'wd', 'ws'
	]
	sns.set(font_scale = 0.5)
	pg = sns.pairplot(data[cols], height = 1.0, aspect = 0.8, plot_kws = dict(linewidth = 1e-3, edgecolor = 'b', s = 0.2),
	                  diag_kind = "kde", diag_kws = dict(shade = True))
	plt.tight_layout()
	plt.savefig(os.path.join(proj_dir, 'graph/pollutants_weather_pair_plot.png'), dpi = 450)
	
	# %% 异常点检测.



