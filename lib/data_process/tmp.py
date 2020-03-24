# -*- coding: utf-8 -*-
"""
Created on 2020/3/24 15:32

@Project -> File: algorithm-tools -> tmp.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import pandas as pd
import sys, os

sys.path.append('../..')

from lib import proj_dir

data_raw = pd.read_csv(os.path.join(proj_dir, 'data/raw/weather/raw_records.csv'))
data_denoised = pd.read_csv(os.path.join(proj_dir, 'data/runtime/data_denoised.csv'))



