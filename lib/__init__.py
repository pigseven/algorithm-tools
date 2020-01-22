# -*- coding: utf-8 -*-
"""
Created on 2020/1/21 下午3:01

@Project -> File: pollution-forecast-offline-training-version-2 -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 初始化
"""

import copy
import sys
import os

sys.path.append('../')

from mod.config.config_loader import config

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap


def search_file_in_dir(dir_path, target_names_list = None):
	"""在目录dir_path中搜索含有target_names_list名单中元素的所有文件"""
	if target_names_list is None:
		return []
	else:
		all_files_names = [p for p in os.listdir(dir_path)]
		target_files = []
		for name in target_names_list:
			target_files += [p for p in all_files_names if name in p]
		target_files = list(set(target_files))
		return target_files



