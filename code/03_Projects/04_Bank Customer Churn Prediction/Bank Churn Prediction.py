# --*-- coding:utf-8 --*--
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
@File: Bank Churn Prediction.py
@Author: Hayes Wong
@Email: 616132717@qq.com
@Time:  2021-1-20 11:19
'''

bank_data = pd.read_csv('../../dataset/BankChurners.csv')
bank_data = bank_data[bank_data.columns[:-2]]
print(bank_data.head())