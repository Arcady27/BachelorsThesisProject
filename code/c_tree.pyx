#encoding: utf-8
# cython: profile=False
import numpy as np
cimport numpy as cnp
#cimport numpy as np
import pickle 
import ast 
from libc.stdlib cimport malloc, free
from copy import deepcopy
from libc.stdlib cimport rand,atoi,atof
from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fprintf, getline, printf
from libc.string cimport strtok, strcpy, memset


cdef extern from "stdlib.h":
	double drand48()
	void srand48(long int seedval)

cdef extern from "time.h":
	long int time(int)

cdef class Node(object):

	cdef public :
		int feature
		int sign
		double quantile
		double threshold
		int class_left
		int class_right

	cdef void print_node(self):
		print self.feature,self.sign,self.threshold,self.classes[0],self.classes[1]
		
	def __init__(self,feature_,sign_,quantile_,threshold_,class0,class1):
		self.feature = feature_
		self.sign = sign_
		self.quantile = quantile_
		self.threshold = threshold_
		self.class_left = class0
		self.class_right = class1
			
cdef class Tree(object):


	cdef public:
		int max_depth
		int num_nodes
		int NUM_FEATURES
		int NUM_CLASSES
		Node[:] genom

	cdef double *obj

	def __init__(self, max_depth, num_features, num_classes, genom=None):
		cdef:
			int i
			int feature
			int sign
			int quantile
			double threshold

		srand48(100)

		self.max_depth = max_depth
		self.num_nodes = np.power(2,self.max_depth)-1

		self.NUM_FEATURES = num_features
		self.NUM_CLASSES = num_classes

		#self.obj =  <double *>malloc(self.NUM_FEATURES * sizeof(double))
		#generate random tree
		addition_classes = np.array([None] * self.num_nodes)
		self.genom = addition_classes

		for i in xrange(self.num_nodes):
			feature = rand()%self.NUM_FEATURES
			sign = rand()%2
			quantile = rand()%100
			threshold = 0
			classes = np.random.permutation(np.arange(self.NUM_CLASSES))[0:2]	

			addition_classes[i] = Node(feature_=feature, sign_=sign, quantile_=quantile, threshold_=threshold, class0=classes[0], class1=classes[1])

	cdef void print_tree_cython(self):
		cdef:
			int d
			int first
			int last
			int cur_power

		cur_power = 1
		print 'PRINTING TREE'
		
		for d in xrange(self.max_depth):
			first = cur_power - 1
			last = cur_power*2 - 1
			for i in xrange(first, last):
				print self.genom[i].feature, self.genom[i].sign, self.genom[i].quantile, self.genom[i].threshold, self.genom[i].class_left,self.genom[i].class_right,'       ',
			print '\n',

			cur_power *= 2

	cdef void print_tree_file_cython(self,filename):
		cdef:
			int d
			int first
			int last
			int cur_power
			FILE* cfile = fopen(filename, "w")
		
		cur_power = 1
		#print 'PRINTING TREE'
		
		for d in xrange(self.max_depth):
			first = cur_power - 1
			last = cur_power*2 - 1
			for i in xrange(first, last):
				fprintf(cfile,"%d %d %f %f %d %d\t", self.genom[i].feature, self.genom[i].sign, self.genom[i].quantile,self.genom[i].threshold, self.genom[i].class_left,self.genom[i].class_right)
			fprintf(cfile,"\n",0)

			cur_power *= 2

		fclose(cfile)

	cdef void read_tree_file_cython(self,filename):
		cdef:
			int d
			int i
			int j
			int first
			int last
			int cur_power
			FILE* cfile = fopen(filename, "r")
			cdef char* line
			cdef char* node
			cdef char* node_buf = <char*> malloc(1000*sizeof(char))
			cdef char* number
			cdef size_t l = 0

		cur_power = 1
		
		for d in xrange(self.max_depth):
			first = cur_power - 1
			last = cur_power*2 - 1

			getline(&line, &l, cfile)
			#printf ("%s\n", line)
			number = strtok(line," ")
			
			i = first
			j = 0

			
			while number != NULL and i <= last-1:
				#printf ("%d   %s\n",j,number)
				#printf("%d ",i)
	
				if j==0:
					self.genom[i].feature = atoi(number)
				if j==1:
					self.genom[i].sign = atoi(number)
				if j==2:		
					self.genom[i].quantile = atof(number)
				if j==3:		
					self.genom[i].threshold = atof(number)
				if j==4:		
					self.genom[i].class_left = atoi(number)
				if j==5:		
					self.genom[i].class_right = atoi(number)
					
				
				j += 1
				if j == 6:		
					j = 0
					i += 1
				if j ==5:
					number = strtok(NULL, "\t")
				else:
					number = strtok(NULL, " ")

			cur_power *= 2

		fclose(cfile)

	cdef int predict_cython(self, x_):
		#0.18
		cdef:
			int cur_index = 0
			int cur_prediction = 0
			int d
			int depth = self.max_depth
			int first_this_level = 0
			int first_next_level = 0
			int cur_level_index
			int left_son
			int current_power = 1
			int ind
			double thres
			int left = 0
			int right = 0
			double feature_value
		
		
		#0.2

		for d in xrange(depth):

			ind = self.genom[cur_index].feature
			feature_value = x_[ind]

			first_this_level = current_power - 1
			cur_level_index = cur_index - first_this_level
			first_next_level = current_power*2 - 1
			left_son = first_next_level + 2*cur_level_index
			
			current_power *= 2

			thres = self.genom[cur_index].threshold
			if d == self.max_depth - 1:
				left = self.genom[cur_index].class_left
				right = self.genom[cur_index].class_right
			
			if self.genom[cur_index].sign == 0:

				if feature_value <= thres:
					cur_prediction = left
					cur_index = left_son

				else:
					cur_prediction = right
					cur_index = left_son + 1
			else:
				if feature_value > thres:
					cur_prediction = left
					cur_index = left_son
				else:
					cur_prediction = right
					cur_index = left_son + 1

		return cur_prediction

		
	def get_genom_length(self):
		return self.num_nodes

	def get_genom(self):
		print self.genom[0].feature
		r=np.asarray(self.genom)
		print r[0].feature
		return r

	def set_genom(self, genom_):
		cdef int i
		for i in xrange(self.num_nodes):
			
			self.genom[i].feature = genom_[i].feature
			self.genom[i].sign = genom_[i].sign
			self.genom[i].quantile = genom_[i].quantile
			self.genom[i].threshold = genom_[i].threshold
			self.genom[i].class_left = genom_[i].class_left
			self.genom[i].class_right = genom_[i].class_right

	def set_threshold(self, ind, threshold_):
		cdef int i = ind
		self.genom[i].threshold = threshold_

	def set_feature(self, ind, feature_):
		cdef int i = ind
		self.genom[i].feature = feature_

	def get_feature(self,ind):
		cdef int i = ind
		return self.genom[i].feature

	def get_quantile(self,ind):
		cdef int i = ind
		return self.genom[i].quantile

	def predict(self,x_):
		return self.predict_cython(x_)

	def print_tree(self):
		self.print_tree_cython()

	def read_tree(self, fname):
		self.read_tree_file_cython(fname)

	def write_to_file(self,fname):
		self.print_tree_file_cython(fname)
# python setup.py build_ext --inplace
