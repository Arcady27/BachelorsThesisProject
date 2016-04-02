import numpy as np
from copy import copy, deepcopy
import operator
import os 

import c_tree
from c_tree import Tree
import cPickle as pickle

from subprocess import call
import argparse
from numpy import inf


from simulation import Simulation
from calc_features import calc_features


TICKER = ''
INSTRUMENT = 'SANDP-500'
NUM_FEATURES=50
PATH = 'features/'
MAX_POS=5
PERIOD_TRAIN = ('100101','150101')
PERIOD_TEST = ('150101','160101')
MAX_PERIOD_LENGTH = 200000

###############################################################################################

class evolution():

    def __init__(self, tree_depth, num_classes, generation_size, num_generations,keep_alive, feature_change_prob, sign_change_prob, quantile_bias,verbose,class_change_prob,trees_path):

        self.TREES_PATH = trees_path + INSTRUMENT + '/'
        if not os.path.exists(self.TREES_PATH):
            os.makedirs(self.TREES_PATH)
        
        self.depth = tree_depth
        
        self.VERBOSE = verbose
        self.NUM_CLASSES = num_classes
        self.MAX_POS = 1
        #evolution parameters
        self.GENERATION_SIZE = generation_size
        self.NUM_GENERATIONS = num_generations
        self.THRESHOLD = int(keep_alive*self.GENERATION_SIZE)

        ##mutation params
        self.FEATURE_CHANGE_PROB = feature_change_prob
        self.CLASS_CHANGE_PROB = class_change_prob
        self.SIGN_CHANGE_PROB = sign_change_prob
        self.QUANTILE_BIAS = quantile_bias

        self.read_features_period(PERIOD_TRAIN[0], PERIOD_TRAIN[1], 'train')
        self.read_features_period(PERIOD_TEST[0], PERIOD_TEST[1], 'test')
        self.NUM_FEATURES = self.features_train.shape[1]
        
        self.percentiles = np.zeros((self.NUM_FEATURES, 101), dtype = np.float32)
        for i in xrange(self.NUM_FEATURES):
            for j in xrange(101):
                self.percentiles[i,j] = np.percentile(self.features_train[:, i], j)
            print self.percentiles[i,:]

        self.simulation = Simulation()
        self.simulation.load_prices('data/' + TICKER + INSTRUMENT + '_' + PERIOD_TRAIN[0] + '_' + PERIOD_TRAIN[1] + '.txt', mode = 'train')
        self.simulation.load_prices('data/' + TICKER + INSTRUMENT + '_' + PERIOD_TEST[0] + '_' + PERIOD_TEST[1] + '.txt', mode = 'test')


        self.generation = [None for x in xrange(self.GENERATION_SIZE)]
        self.new_generation = [None for x in xrange(self.GENERATION_SIZE)]
        self.generation_scores = [0 for x in xrange(self.GENERATION_SIZE)]
        self.generate_random_generation()

        for i in xrange(self.NUM_GENERATIONS):
            print 'CURRENT GENERATION %d' % (i)
            self.create_new_generation()
            print 'TEST SCORE %d' % (self.calc_score(0, mode = 'test'))
            #if self.generation_scores[0] >= 100:
            #    break

        
        for i in xrange(self.THRESHOLD):
            print self.generation_scores[i],' ',
            self.generation[i].write_to_file(self.TREES_PATH+'tree_'+str(i)+'_'+str(int(self.generation_scores[0])))
        
        print '\n'

        

    def generate_random_generation(self):
        for i in xrange(self.GENERATION_SIZE):
            print i
            self.generation[i] = Tree(max_depth=self.depth, num_classes=self.NUM_CLASSES, num_features=self.NUM_FEATURES)
            self.new_generation[i] = Tree(max_depth=self.depth, num_classes=self.NUM_CLASSES, num_features=self.NUM_FEATURES)

            genom_length = self.generation[i].get_genom_length()
            for j in xrange(genom_length):

                new_threshold = self.percentiles[self.generation[i].genom[j].feature, self.generation[i].genom[j].quantile]
                self.generation[i].set_threshold(j, new_threshold)

            if self.VERBOSE:
                self.generation[i].print_tree()

            self.generation_scores[i] = self.calc_score(i)
        
    
    def create_new_generation(self):
        global DAYS_TRAIN
        
        #DAYS_TRAIN = np.random.permutation(DAYS_TRAIN)

        self.generation_scores, self.generation = (list(t) for t in zip(*sorted(zip(self.generation_scores, self.generation), reverse=True)))
        for i in xrange(self.THRESHOLD):
            print self.generation_scores[i],' ',
        #self.generation[0].write_to_file(self.TREES_PATH+'tree_'+str(i)+'_'+str(int(self.generation_scores[0])))
        print '\n'

        #for i in xrange(self.GENERATION_SIZE):
        #   self.new_generation[i].set_genom(self.generation[i].genom)
        
        #self.generation_scores[0] = self.calc_score(0)
            
        for i in xrange(self.THRESHOLD,self.GENERATION_SIZE):
            print i

            parent1 = np.random.randint(self.THRESHOLD)
            parent2 = np.random.randint(self.THRESHOLD)

            if self.VERBOSE:
                print 'PARENT A: %d' % parent1
                self.generation[parent1].print_tree()
                print 'PARENT B: %d' % parent2
                self.generation[parent2].print_tree()

            child = self.make_child_subtrees(self.generation[parent1], self.generation[parent2])
            self.generation[i].set_genom(child.genom)
                
            if self.VERBOSE:
                print 'child before mutation'
                self.generation[i].print_tree()


            self.mutate(child.genom)

            self.generation[i].set_genom(child.genom)
            #self.generation_scores[new_ind] = self.calc_score(new_ind)

            if self.VERBOSE:
                print 'child after mutation'
                self.generation[i].print_tree()
                print '\n\n'

            self.generation_scores[i] = self.calc_score(i)
            print self.generation_scores[i]

        #for i in xrange(1,self.GENERATION_SIZE):
        #   self.generation[i].set_genom(self.new_generation[i].genom)
        
    
    def copy_gen(self, genom_a, genom_b, i):
        genom_a[i].feature = genom_b[i].feature
        genom_a[i].sign = genom_b[i].sign
        genom_a[i].quantile = genom_b[i].quantile
        genom_a[i].threshold = genom_b[i].threshold
        genom_a[i].class_left = genom_b[i].class_left
        genom_a[i].class_right = genom_b[i].class_right
        return genom_a

    def make_child_random(self, parent_a, parent_b):
        
        child = Tree(max_depth=parent_a.max_depth, num_classes=parent_a.NUM_CLASSES, num_features=parent_a.NUM_FEATURES)

        for i in xrange(parent_a.get_genom_length()):
            alpha = np.random.uniform(0, 1)
            if alpha > 0.5:
                child.genom = self.copy_gen(child.genom, parent_b.genom, i)             
            else:
                child.genom = self.copy_gen(child.genom, parent_a.genom, i)             
            
        return child

    def make_child_subtrees(self, parent_a, parent_b):

        child = Tree(max_depth=parent_a.max_depth, num_classes=parent_a.NUM_CLASSES, num_features=parent_a.NUM_FEATURES)
        for i in xrange(parent_a.get_genom_length()):
            child.genom = self.copy_gen(child.genom, parent_a.genom, i)

        root_depth = np.random.randint(self.depth)
        current_power = np.power(2,root_depth)
        first_this_level = current_power - 1
        first_next_level = current_power*2 - 1
        root_index = np.random.randint(first_this_level, first_next_level)

        if self.VERBOSE:
            print 'ROOT INDEX: %d' % (root_index)
        
        min_index = root_index
        max_index = root_index+1

        child.genom = self.copy_gen(child.genom, parent_b.genom, root_index)

        for d in xrange(root_depth, self.depth - 1):

            first_this_level = current_power - 1
            first_next_level = current_power*2 - 1
            current_power *= 2

            #find left_son of min_index
            cur_level_index = min_index - first_this_level
            left_son = first_next_level + 2*cur_level_index

            #find right_son of max_index
            cur_level_index = max_index-1 - first_this_level
            right_son = first_next_level + 2*cur_level_index + 1

            min_index = left_son
            max_index = right_son+1
            for i in xrange(min_index, max_index):
                child.genom = self.copy_gen(child.genom, parent_b.genom, i)

        return child

    def mutate(self, genom):

        for i in xrange(len(genom)):

            change_class = np.random.uniform(0, 1)
            if change_class < self.CLASS_CHANGE_PROB:
                genom[i].class_left = np.random.randint(0, self.NUM_CLASSES)
                genom[i].class_right = (genom[i].class_left + np.random.randint(1, self.NUM_CLASSES) ) % self.NUM_CLASSES


            change_feature = np.random.uniform(0, 1)
            if change_feature < self.FEATURE_CHANGE_PROB:
                genom[i].feature = np.random.randint(0, self.NUM_FEATURES)

            else:
                change_sign = np.random.uniform(0, 1)
                if change_sign < self.SIGN_CHANGE_PROB:
                    genom[i].sign = (genom[i].sign + 1) % 2

                z = np.random.randint(-self.QUANTILE_BIAS, self.QUANTILE_BIAS)
                genom[i].quantile += z
                if genom[i].quantile < 0:
                    genom[i].quantile = 0
                elif genom[i].quantile > 100:
                    genom[i].quantile = 100

            genom[i].threshold = self.percentiles[genom[i].feature, genom[i].quantile]

        return 0

    def calc_score(self, tree_index, mode = 'train'):
        global MAX_POS
        global actions,results


        tree = self.generation[tree_index]
        
        if mode == 'train':
            
            for i in xrange(self.features_train.shape[0]):

                obj = self.features_train[i, :]
                actions[i] = tree.predict(obj)
                #print actions[i]
            result, deals = self.simulation.run_simulation(mode = 'train', actions = actions, max_pos = self.MAX_POS)
            
        if mode == 'test':
            
            for i in xrange(self.features_test.shape[0]):

                obj = self.features_test[i, :]
                actions[i] = tree.predict(obj)

            result, deals = self.simulation.run_simulation(mode = 'test', actions = actions, max_pos = self.MAX_POS)
        
        print 'result = %f, deals = %d' % (result, deals)

        return result

    def read_features_period(self, period_start, period_end, mode):
        global PATH, TICKER, INSTRUMENT
        global features_day

        filename_features = PATH + TICKER + INSTRUMENT+ '_' + period_start + '_' + period_end + '_features.txt'
        
        try:
            features_day = np.genfromtxt(filename_features, delimiter=',')
        except Exception:
            calc_features('data/' + TICKER + INSTRUMENT + '_' + period_start + '_' + period_end + '.txt')
            features_day = np.genfromtxt(filename_features, delimiter=',')

        print features_day.shape
        print features_day

        if mode == 'train':
            self.features_train = features_day
        else:
            self.features_test = features_day
        

def main():

    global features, instrument, actions,results

    actions = np.zeros((200000,), dtype=np.int32)
        
    evo=evolution(tree_depth=2, num_classes=3, generation_size=50, num_generations=100, keep_alive=10.0/50, 
            feature_change_prob=0.3, sign_change_prob=0.3, class_change_prob=0.2,quantile_bias=30, verbose=True,trees_path='trees/')

#import cProfile
#cProfile.run('main()')
main()
