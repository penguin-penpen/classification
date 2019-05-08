# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as pl
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

# myarray = np.fromfile('usagov.txt')
# myarray = np.loadtxt('usagov.txt')
# myarray = np.fromfile('iris.csv')
raw = pd.read_csv('new.csv')
data = raw.values
print (data)
x = data[:,3]
# print x
y = data[:,6]
# z = data[:,2]
# w = data[:,3]
color_dict = {'优': 'b',
              '良': 'g',
               '轻度污染':'tomato',
              '中度污染':'silver',
              '重度污染':'r'}

pl.scatter(x,y,color = [color_dict[i] for i in data[:,2]])

plt.legend(loc = 'best')
#pl.figure()
plt.xlabel(u"PM2.5") #X轴标签
plt.ylabel("一氧化碳") #Y轴标签
plt.title("数据散点图") #标题
pl.show()
s = set(data[:,2])
#print (s)