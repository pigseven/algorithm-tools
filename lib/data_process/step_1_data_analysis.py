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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import sys, os

sys.path.append('../..')

from lib import proj_dir


if __name__ == '__main__':
	# %% 载入数据.
	from lib.data_process.tmp import data_denoised as data
	cols = [
		'pm10', 'pm25', 'o3', 'so2', 'co', 'no2', 'aqi',
		'clock_num', 'weekday', 'month', 'sd', 'weather', 'temp', 'wd', 'ws'
	]
	
	# %% pairplot作图.
	# sns.set(font_scale = 0.5)
	# pg = sns.pairplot(data[cols], height = 1.0, aspect = 0.8, plot_kws = dict(linewidth = 1e-3, edgecolor = 'b', s = 0.2),
	#                   diag_kind = "kde", diag_kws = dict(shade = True))
	# plt.tight_layout()
	# plt.savefig(os.path.join(proj_dir, 'graph/pollutants_weather_pair_plot.png'), dpi = 450)
	
	# %% 异常点检测.
	# TODO: 异常结果可视化.
	isoforest = IsolationForest(n_estimators = 100, max_samples = 0.9)
	X_train = np.array(data[cols])
	
	idxs = list(range(X_train.shape[0]))
	random.shuffle(idxs)
	X_train = X_train[idxs[: 5000], :]
	
	isoforest.fit(X_train)
	y_pred_train = isoforest.predict(X_train)
	
	scores = isoforest.decision_function(X_train)
	
	# 异常识别可视化.
	# tsne = TSNE()
	# embeddings = tsne.fit_transform(X_train)  # 进行数据降维, 降成两维
	# plt.scatter(embeddings[:, 0], embeddings[:, 1], s = 6, c = y_pred_train)
	
	


