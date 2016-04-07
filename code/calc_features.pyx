#encoding: utf-8
#cython: p#rofile=True

import numpy as np
import pandas as pd
import csv
from collections import Counter,deque
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libc.string cimport memset
from libc.math cimport sqrt
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, getline, printf

cdef:
	int NUM_FEATURES_INDEP = 5 
	int NUM_FEATURES_DEP = 14
	int PERIODS = 3
	int SIZE
	double EPS = 0.00000001

	vector[double*] features
	double open_arr[2000000]
	double close_arr[2000000]
	double high_arr[2000000]
	double low_arr[2000000]
	double vol_arr[2000000]
	int periods[3]

periods[:] = [10,60,240]

cdef void read_features(filename):
	global SIZE,open_arr,_arr,high_arr,low_arr, vol_arr
	cdef:
		int i

	features = pd.read_csv(filename)
	print features
	del features['<TICKER>']
	del features['<PER>'] 
	del features['<DATE>'] 
	del features['<TIME>'] 

	features.columns = ['open','high','low','close','vol']
	SIZE = features.shape[0]
	print SIZE
	for i in xrange(SIZE):
		open_arr[i] = features.open[i]
		close_arr[i] = features.close[i]
		high_arr[i] = features.high[i]
		low_arr[i] = features.low[i]
		vol_arr[i] = features.vol[i]
	print features

cdef double mean(double *arr, int s, int f):
	cdef:
		int i, n = f-s
		double sum_ = 0.0
	for i in xrange(s,f):
		sum_ += arr[i]
	return sum_/n

cdef double stdev(double *arr, int s, int f):
	cdef:
		int i, n = f-s
		double std = 0.0, mean = 0.0

	for i in xrange(s,f):
		mean += arr[i]
	
	for i in xrange(s,f):
		std += (arr[i] - mean)*(arr[i] - mean)
	
	std = sqrt(std/(n-1))
	return std

cdef double max(double *arr, int s, int f):
	cdef:
		int i
		double max_ = -1000000000
	for i in xrange(s,f):
		if arr[i] > max_:
			max_ = arr[i]
	
	return max_

cdef double min(double *arr, int s, int f):
	cdef:
		int i
		double min_ = 1000000000
	for i in xrange(s,f):
		if arr[i] < min_:
			min_ = arr[i]
	
	return min_

cdef int sign(double x):
	if x > 0:
		return 1
	if x < 0:
		return -1
	return 0

cdef void calc_features_cython(filename):
	global NUM_FEATURES_DEP, NUM_FEATURES_INDEP, PERIODS, SIZE, EPS
	global open_arr,close_arr,high_arr,low_arr, vol_arr, features
	
	filename_out = filename.replace('data','features').replace('.txt','')+'_features.txt'
	print filename_out
	print PERIODS, SIZE
	cdef:
		int p,i,j,q
		double open_= 0.0,close = 0.0,high= 0.0, low= 0.0, vol = 0.0,close_prev = 0.0
		double sma = 0.0, std = 0.0, highest_high = 0.0, lowest_low = 0.0

		FILE* cfile = fopen(filename_out, "w")

	read_features(filename)
	
	print PERIODS,SIZE
	cdef:
		double *smas = <double *>malloc(PERIODS * sizeof(double))
		double *true_range = <double *>malloc(SIZE * sizeof(double))
		
	print SIZE
	for p in xrange(PERIODS*NUM_FEATURES_DEP + NUM_FEATURES_INDEP):
		features.push_back(<double *>malloc(SIZE * sizeof(double)))		
		
	for i in xrange(SIZE): 
		if i%1000==0:
			print i

		#print i
		for j in xrange(NUM_FEATURES_INDEP + NUM_FEATURES_DEP*PERIODS):
			features[j][i] = 0.0

		for j in xrange(PERIODS):
			

			t = periods[j]
			q = j*NUM_FEATURES_DEP
			
			close = close_arr[i]
			close_prev = 0.0
			if i>=1:
				close_prev = close_arr[i-1]         
			open_ = open_arr[i]
			high = high_arr[i]
			low = low_arr[i]
			vol = vol_arr[i]
			
			sma = mean(close_arr, i-t, i+1)
			smas[j] = sma
			std = stdev(close_arr, i-t, i+1)
			highest_high = max(high_arr, i-t, i+1)
			lowest_low = min(low_arr, i-t, i+1)
			
			#print close, vol, highest_high, lowest_low ,t, close_prev, high, low, open_
			
			true_range[i] = np.max([high - low, high - close_prev, close_prev - low])
			if i>=t:
				
				features[NUM_FEATURES_INDEP + q][i] = (close - close_arr[i-t]) / (close_arr[i-t] + EPS) #proc
				features[NUM_FEATURES_INDEP + 1 + q][i] = (vol - vol_arr[i-t]) / (vol_arr[i-t] + EPS) #vroc
				features[NUM_FEATURES_INDEP + 2 + q][i] = close_arr[i-t] - close          #momentum
				features[NUM_FEATURES_INDEP + 3 + q][i] = features[NUM_FEATURES_INDEP + 2 + q][i-t] - features[NUM_FEATURES_INDEP + 2 + q][i] #Price acceleration
				features[NUM_FEATURES_INDEP + 4 + q][i] = mean(true_range, i-t, i+1) / close #natr
				
				features[NUM_FEATURES_INDEP + 5 + q][i] = -100*((highest_high - close) / (highest_high - lowest_low + EPS)) #williams %r

				features[NUM_FEATURES_INDEP + 6 + q][i] = features[NUM_FEATURES_INDEP + 6 + q][i-1] + vol * sign(close - close_prev) #OBV
				
				features[NUM_FEATURES_INDEP + 7 + q][i] = ((close - lowest_low) / (highest_high - lowest_low + EPS))*100 #stoch
				features[NUM_FEATURES_INDEP + 8 + q][i] = mean(features[NUM_FEATURES_INDEP + 7 + q], i-t, i+1) #F%K
				features[NUM_FEATURES_INDEP + 9 + q][i] = mean(features[NUM_FEATURES_INDEP + 8 + q], i-3, i+1) #S%K
				features[NUM_FEATURES_INDEP + 10 + q][i] = mean(features[NUM_FEATURES_INDEP + 9 + q], i-3, i+1) #S%D
				
				features[NUM_FEATURES_INDEP + 11 + q][i] = std / sqrt(<double>(t)) #volatility
				
				middle_band = sma
				upper_band = middle_band + 2*std
				lower_band = middle_band - 2*std 
			
				features[NUM_FEATURES_INDEP + 12 + q][i] = (close - lower_band) / (upper_band - lower_band) #b%
				features[NUM_FEATURES_INDEP + 13 + q][i] = upper_band - lower_band #bwidth

			
		if i >= 1:
			features[0][i] = features[0][i-1] + vol * ((close - close_prev) / close_prev) #PVT

		features[1][i] = vol * ((close - low) - (high - close)) / (high - low + EPS) #AD
		features[2][i] += 100 * ((high - open_) + (close - low)) / ((high - low + EPS) * 2) #accumulation distribution oscillator
		features[3][i] = ((smas[0] - smas[PERIODS - 1]) / smas[PERIODS - 1]) * 100 #PPO
		features[4][i] = smas[0] - smas[PERIODS - 1] #MACD

		for j in xrange(NUM_FEATURES_INDEP + NUM_FEATURES_DEP*PERIODS):
			fprintf(cfile,"%f,",features[j][i])
		fprintf(cfile,"%f\n",features[j][i])

	fclose(cfile)
	free(smas)
	free(true_range)
	#for i in xrange(PERIODS*NUM_FEATURES_DEP + NUM_FEATURES_INDEP):
	#	free(features[i])

def calc_features(filename):
	calc_features_cython(filename)