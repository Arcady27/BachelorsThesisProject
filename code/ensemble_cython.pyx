import numpy as np
from copy import copy, deepcopy
import operator

import c_tree
from c_tree import Tree
import cPickle as pickle

from subprocess import call
import argparse
from numpy import inf
from random import shuffle 
from libcpp.vector cimport vector
from os import listdir
from os.path import isfile, join

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from simulation import Simulation
from calc_features import calc_features


TICKER = ''
INSTRUMENT = 'LKOH'
NUM_FEATURES=50
PATH = 'features/'
MAX_POS=5
PERIOD_TRAIN = ('120101','140101')
PERIOD_TEST = ('140101','160101')
FREQUENCY = '10minutes'
MAX_PERIOD_LENGTH = 200000
###############################################################################################

cdef class evolution(object):
	cdef:
		int MAX_WEIGHT
		int VERBOSE
		int depth
		int NUM_CLASSES
		int NUM_FEATURES
		int GENERATION_SIZE
		int NUM_GENERATIONS
		int THRESHOLD
		int GENOM_LENGTH
		int NUM_TREES
		int MAX_POS
		int COMI
		double MUTATION_PROB
		int* predictions_train
		int*  predictions_test
		
		int*  weights
		results
	cdef:
		features_train, features_test, simulation

	cdef public:
		trees, X, generation, generation_scores, PATH, files

	def __init__(self, tree_depth, num_classes, keep_alive, generation_size, num_generations,verbose, mutation_prob, max_weight,folder, comi):

		self.depth = tree_depth - 1
		
		self.VERBOSE = verbose
		self.NUM_CLASSES = num_classes
		
		self.MAX_POS = 1
		self.COMI = comi
		self.MAX_WEIGHT = max_weight
		self.PATH = folder
		#evolution parameters
		self.GENERATION_SIZE = generation_size
		self.NUM_GENERATIONS = num_generations
		self.THRESHOLD = int(keep_alive*self.GENERATION_SIZE)

		##mutation params
		self.MUTATION_PROB = mutation_prob

		self.read_features_period(PERIOD_TRAIN[0], PERIOD_TRAIN[1], 'train')
		self.read_features_period(PERIOD_TEST[0], PERIOD_TEST[1], 'test')
		self.NUM_FEATURES = self.features_train.shape[1]
		print 'TRAIN shape: %d TEST shape %d' % (self.features_train.shape[0],self.features_test.shape[0])

		self.trees = []
		self.NUM_TREES = 0
		self.read_trees()
		self.GENOM_LENGTH = self.NUM_TREES
		print self.files
		
		self.results = np.zeros((self.NUM_GENERATIONS,6), dtype=np.float32)
		results_ = np.fromfile('results/results_' + INSTRUMENT + '_' + FREQUENCY + '_' + str(self.depth + 1) + '.txt',dtype = np.float32)
		results_ = results_.reshape(results_.shape[0]/6,6) 
		self.results[0:results_.shape[0],:] = results_
		self.results[:,4] = self.results[results_.shape[0]-1, 4]
		self.results[:,5] = self.results[results_.shape[0]-1, 5]
		
		self.weights = <int *>malloc(self.GENOM_LENGTH * sizeof(int))
		self.predictions_train = <int *>malloc(self.features_train.shape[0] * self.NUM_TREES * sizeof(int))
		self.predictions_test = <int *>malloc(self.features_test.shape[0] * self.NUM_TREES * sizeof(int))
			
		self.calc_predictions()

		self.simulation = Simulation()
		self.simulation.load_prices('data/' + TICKER + INSTRUMENT + '_' + PERIOD_TRAIN[0] + '_' + PERIOD_TRAIN[1] +  '_' + FREQUENCY + '.txt', mode = 'train')
		self.simulation.load_prices('data/' + TICKER + INSTRUMENT + '_' + PERIOD_TEST[0] + '_' + PERIOD_TEST[1] +  '_' + FREQUENCY + '.txt', mode = 'test')

		self.generation = [np.random.random_integers(0,self.MAX_WEIGHT,(self.GENOM_LENGTH,)) for x in xrange(self.GENERATION_SIZE)]
		self.generation_scores = np.zeros((self.GENERATION_SIZE,),dtype=np.float32)
		
	
		np.array(self.generation[0]).tofile('ensembles/best_ensemble_'+str(self.generation_scores[0])+'_'+str(-10000))

		for i in xrange(len(self.generation)):
			for j in xrange(self.GENOM_LENGTH):
				self.generation[i][j] = np.random.randint(0,self.MAX_WEIGHT)

		print self.generation
		
		for i in xrange(self.GENERATION_SIZE):
			self.generation_scores[i] = self.calc_score(i)
		print self.generation_scores

		self.generation = list(self.generation)
		print len(set(self.generation_scores))
		scores_set = []
		gen = []
		for i in xrange(len(self.generation_scores)):
			score = self.generation_scores[i]
			if score not in scores_set:
				scores_set.append(score)
				gen.append(self.generation[i])

		gen = np.array(gen)
		print gen.shape
		self.generation = gen
		self.generation_scores = scores_set
		self.GENERATION_SIZE = gen.shape[0]
		self.THRESHOLD = int(keep_alive*self.GENERATION_SIZE)

		print sorted(zip(self.generation_scores, self.generation), reverse = True)


		for i in xrange(self.NUM_GENERATIONS):
			print "NUM GENERATION %d" % (i)
			self.create_new_generation()
			#if self.MUTATION_PROB > 0.05:
			self.MUTATION_PROB *= 1.01
			
			test_score = self.calc_score(0, mode = 'test')
			print 'TEST SCORE %d' % (test_score)
			
			if i < self.results.shape[0]:
				self.results[i, 0] = self.generation_scores[0]
				self.results[i, 1] = test_score
				#print self.results

		self.results[0:i, :].tofile('results/results_ensemble_' + INSTRUMENT + '_' + FREQUENCY + '_' + str(self.depth+1) + '.txt')

		for i in xrange(self.THRESHOLD):
			print self.generation_scores[i],' ',
		print '\n'

		
	def read_trees(self):

		for path in self.PATH:
			
			path = path + INSTRUMENT + '/' + FREQUENCY + '/' + str(self.depth+1) + '/'

			self.files = [f for f in listdir(path) if isfile(join(path, f))]
			self.files=sorted(self.files)

			for i in xrange(len(self.files)):
				filename = self.files[i]
				self.trees.append(None)
				self.trees[self.NUM_TREES] = Tree(max_depth=self.depth, num_classes=self.NUM_CLASSES, num_features=self.NUM_FEATURES)
				print filename
				self.trees[self.NUM_TREES].read_tree(path+'/'+filename)
				if self.VERBOSE:
					self.trees[self.NUM_TREES].print_tree()
				self.NUM_TREES += 1

	def create_new_generation(self):
		
		self.generation_scores, self.generation = (list(t) for t in zip(*sorted(zip(self.generation_scores, self.generation), reverse=True)))
		for i in xrange(self.THRESHOLD):
			print self.generation_scores[i],' ',
		print '\n'

		for i in xrange(min(10,self.GENERATION_SIZE)):
			np.array(self.generation[i]).tofile('ensembles/ensemble_'+str(int(self.generation_scores[i])))

		for i in xrange(self.THRESHOLD,self.GENERATION_SIZE):
			print i
			#print '\nRES TEST: %d\n' % (res_test)


			parent1 = np.random.randint(self.THRESHOLD)
			parent2 = np.random.randint(self.THRESHOLD)

			print self.generation[parent1],self.generation[parent2]
			child = self.make_child(self.generation[parent1], self.generation[parent2])
			self.generation[i] = child
			print 'child before mutation:\n',self.generation[i]	
			self.generation[i] = self.mutate(child)
			print 'child after mutation:\n',self.generation[i]	
			
			self.generation_scores[i] = self.calc_score(i)
			print self.generation_scores

			

	def make_child(self, parent_a, parent_b):
		
		child = np.zeros((self.GENOM_LENGTH,),dtype=np.int32)

		for i in xrange(self.GENOM_LENGTH):
			alpha = np.random.uniform(0, 1)
			if alpha > 0.5:
				child[i] = parent_a[i]				
			else:
				child[i] = parent_b[i]				
			
		return child


	def mutate(self, genom):

		for i in xrange(self.GENOM_LENGTH):

			change = np.random.uniform(0, 1)
			if change < self.MUTATION_PROB:
				genom[i] = np.random.randint(0,self.MAX_WEIGHT)
		return genom

	cdef void calc_predictions(self):
		cdef:
			int i
			int c
			int day_num
			int tree
			int ntrees = len(self.trees)

		print 'calculating predictions ...'
		
		for i in xrange(self.features_train.shape[0]):
			if i % 1000 == 0:
				print i
			
			for c in xrange(self.NUM_TREES):
				self.predictions_train[i*self.NUM_TREES+c] = 0

			
				obj = self.features_train[i, :]
				#print obj
				for tree in xrange(ntrees):
					predicted_action = self.trees[tree].predict(obj)
					#if predicted_action!=0:
					#	print predicted_action
					self.predictions_train[i*self.NUM_TREES + tree] = predicted_action
		
		for i in xrange(self.features_test.shape[0]):
			if i % 1000 == 0:
				print i
			
			for c in xrange(self.NUM_TREES):
				self.predictions_test[i*self.NUM_TREES+c] = 0

				obj = self.features_test[i, :]
				#print obj
				for tree in xrange(ntrees):
					predicted_action = self.trees[tree].predict(obj)
					#if predicted_action!=0:
					#	print predicted_action
					self.predictions_test[i*self.NUM_TREES + tree] = predicted_action
		
			#for i in xrange(ntrees):
			#	for j in xrange(all_days_features[i].shape[0]*self.NUM_TREES):
			#		print self.predictions[i][j]


	cdef int cython_argmax(self,int *arr,int n):
		cdef:
			int i
			int max_ind = 0
			double maxi = -100000
		for i in xrange(n):
			if arr[i] > maxi:
				maxi = arr[i]
				max_ind = i
		return max_ind

	cdef double calc_score(self, g,mode='train'):
		global MAX_POS
		global actions,results, all_days_features_test, all_days_features
		
		cdef:
			int votes[3]
			int i
			int c
			int predicted_action
			int day_num
			int weight
			int tree_index
			int num_days
			int gene

		
		for i in xrange(self.GENOM_LENGTH):
			self.weights[i] = self.generation[g][i]
				
		if mode == 'train':
				
			for i in xrange(self.features_train.shape[0]):

				for c in xrange(self.NUM_CLASSES):
					votes[c] = 0
					
				for gene in xrange(self.GENOM_LENGTH):
					weight = self.weights[gene]
					if weight > 0:	
						prediction = self.predictions_train[i*self.NUM_TREES + gene]
						votes[prediction] += weight

				actions[i] = self.cython_argmax(votes,self.NUM_CLASSES)
				#if actions[i]!=0:
				#	print votes[0],votes[1],votes[2],actions[i]

			result, deals = self.simulation.run_simulation(mode = mode, actions = actions, max_pos = self.MAX_POS, comis = self.COMI)
			#print result, deals
		
		elif mode == 'test':
			for i in xrange(self.features_test.shape[0]):

				for c in xrange(self.NUM_CLASSES):
					votes[c] = 0

				for gene in xrange(self.GENOM_LENGTH):
					weight = self.weights[gene]
					if weight > 0:		
						prediction = self.predictions_test[i*self.NUM_TREES + gene]
						votes[prediction] += weight

				actions[i] = self.cython_argmax(votes,self.NUM_CLASSES)
				#if actions[i]!=0:
				#	print votes[0],votes[1],votes[2],actions[i]

			result, deals = self.simulation.run_simulation(mode = mode, actions = actions, max_pos = self.MAX_POS, comis = self.COMI)


		return result + np.random.uniform(0,0.1)

	def read_features_period(self, period_start, period_end, mode):
		global PATH, TICKER, INSTRUMENT, FREQUENCY
		global features_day

		filename_features = PATH + TICKER + INSTRUMENT+ '_' + period_start + '_' + period_end +  '_' + FREQUENCY + '_features.txt'
		
		try:
			features_day = np.genfromtxt(filename_features, delimiter=',')
		except Exception:
			calc_features('data/' + TICKER + INSTRUMENT + '_' + period_start + '_' + period_end +  '_' + FREQUENCY + '.txt')
			features_day = np.genfromtxt(filename_features, delimiter=',')

		print features_day.shape
		print features_day

		if mode == 'train':
			self.features_train = features_day
		else:
			self.features_test = features_day
		


	cdef void finish(self):

		try:
			free(self.predictions_train)
			free(self.predictions_test)
			free(self.weights)
		except Exception:
			pass

def main(depth, comi):
	global features, instrument, actions

	actions = np.zeros((200000,), dtype=np.int32)
	
	evo  = evolution(tree_depth=depth, num_classes=3, generation_size=100, num_generations=10, keep_alive=10.0/50, verbose=True, 
		mutation_prob=0.05,max_weight=10,folder=['trees/'], comi = comi)
	evo.finish()

#import cProfile
#cProfile.run('main()')
periods = ['10minutes','1hour','1day']
instruments = ['GAZP','LKOH','SBER','SPFB.MIX','SPFB.RTS','MICEX','RTSI','GE','NASDAQ.AAPL','XOM','NASDAQ.GOOG','NY.MSFT',
               'C','DSX','JPM','WMT','NQ-100-FUT','SP500']

comis = [0.000035 for x in xrange(len(instruments))]
depths = [3,6]
for i in xrange(len(instruments)):
	instr = instruments[i]
	comi = comis[i]
	INSTRUMENT = instr
	
	for period in periods:
		FREQUENCY = period
		
		for d in depths:
			main(d, comi)