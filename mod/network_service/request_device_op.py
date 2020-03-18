# -*- coding: utf-8 -*-
"""
Created on 2020/2/21 14:47

@Project -> File: realtime-wind-rose-diagram -> request_device_op.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 
"""

import pandas as pd
import requests
import json

req_ops_available = ['pull_device_value']


class RequestOperation(object):
	"""请求操作"""
	
	def __init__(self, req_op = None, params = None):
		"""
		初始化
		"""
		try:
			assert req_op in req_ops_available
		except:
			raise ValueError('ERROR: req_op {} not in req_ops_available {}'.format(req_op, req_ops_available))
		
		if req_op == 'pull_device_value':
			try:
				assert params is not None
			except:
				raise ValueError('ERROR: params is None.')
		else:
			pass
		
		self.req_op = req_op
		self.params = params
		
	def _build_url(self, root_url):
		url = None
		if self.req_op == 'pull_device_value':
			url = root_url
		else:
			pass
		return url
	
	def request(self, root_url):
		url = self._build_url(root_url)
		retry = 0
		while True:
			if self.req_op in ['pull_device_value']:
				resp = requests.get(url, params = self.params)
			else:
				raise ValueError('ERROR: invalid self.req_op {}'.format(self.req_op))
			
			if retry < 3:
				if resp.status_code >= 500:
					# 重试请求.
					print('Retry requesting, retry = {}'.format(retry))
					retry += 1
					continue
				elif resp.status_code in [200, 204]:
					break
			else:
				raise RuntimeError('ERROR: reach max request time = 3, cannot get response, req_op = {}'.format(self.req_op))
		
		data = pd.DataFrame(json.loads(resp.text)['data'])
		return data
	




