# -*- coding: utf-8 -*-
"""
Created on 2019/12/9 下午3:56

@Project -> File: industrial-research-guodian-project -> time_conversion.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时间转换
"""

import time


def time2stamp(t, time_format = '%Y-%m-%d %H:%M:%S'):
	"""时间转为时间戳"""
	stp = int(time.mktime(time.strptime(t, time_format)))
	return stp


def stamp2time(stp, time_format = '%Y-%m-%d %H:%M:%S'):
	t_arr = time.localtime(stp)
	t = time.strftime(time_format, t_arr)
	return t


if __name__ == '__main__':
	t = '2019-06-19 16:00:00'
	stp = time2stamp(t)
	t = stamp2time(stp)



