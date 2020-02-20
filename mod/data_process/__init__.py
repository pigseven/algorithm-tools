# -*- coding: utf-8 -*-
"""
Created on 2020/2/20 15:33

@Project -> File: algorithm-tools -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

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
		raise ValueError('ERROR: list is not sorted by ascending.')
	
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


