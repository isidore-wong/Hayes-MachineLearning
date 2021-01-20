# --*-- coding:utf-8 --*--

'''
@File: utils.py
@Author: Hayes Wong
@Email: 616132717@qq.com
@Time:  2021-1-11 17:05
'''

'''
关于各个算法的基础计算
'''

def sign(x):
    '''
    符号函数
    :param x:
    :return:
    '''
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0