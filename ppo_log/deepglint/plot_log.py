#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:52:09 2019

@author: deepglint
"""

import re
import matplotlib.pyplot as plt
import numpy as np


def main():
    file = open('cal.log','r')



    list = []
    # search the line including accuracy
    count = 0
    reward = 0
    av = 5#38*10
    #av = 560
    #av = 20
    #av = 2240
    for line in file:
        '''
        m=re.search('Train-mse', line)
        if m:
            n=re.search('[0]\.[0-9]+', line) # 正则表达式
            if n is not None:
                list.append(n.group()) # 提取精度数字
        '''
        #print(float(line.split()[0]))
	#if( isinstance(line.split()[0], float) ):
	if('-' in line.split()[0] and line.split()[0][0]!='-'):
	    pass
	else:
	    reward  =reward+ float(line.split()[0])
            count+=1

        if(count%av==0):
            list.append(float(reward)/av)
            count=0
            reward = 0



    file.close()

    plt.plot(list[:])
    #plt.plot(list[680:])
    #plt.plot(list, 'r')
    plt.xlabel('count')
    plt.ylabel('reward')
    plt.title('rl_train')
    plt.show()

if __name__ == '__main__':
    main()
