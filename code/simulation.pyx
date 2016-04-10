import numpy as np
import pandas as pd

cdef class Simulation(object):

	cdef:
		double open_test[200000], close_test[200000], high_test[200000], low_test[200000]
		double open_train[200000], close_train[200000], high_train[200000], low_train[200000]
		int SIZE_TRAIN, SIZE_TEST
		double buy_and_hold_train, buy_and_hold_test
		results
	
	
	def __init__(self, int size_train = 200000, int size_test=100000):
		self.SIZE_TRAIN = size_train
		self.SIZE_TEST = size_test

	cdef void load_prices_cython(self, filename, mode):
		features = pd.read_csv(filename)
		del features['<TICKER>']
		del features['<PER>'] 
		del features['<DATE>'] 
		del features['<TIME>'] 

		features.columns = ['open','high','low','close','vol']
		
		if mode == 'train':
			self.SIZE_TRAIN = features.shape[0]

			for i in xrange(self.SIZE_TRAIN):
				self.open_train[i] = features.open[i]
				self.close_train[i] = features.close[i]
				self.high_train[i] = features.high[i]
				self.low_train[i] = features.low[i]
		else:
			self.SIZE_TEST = features.shape[0]
			self.results = np.zeros((200000,),dtype = np.float32)

			for i in xrange(self.SIZE_TEST):
				self.open_test[i] = features.open[i]
				self.close_test[i] = features.close[i]
				self.high_test[i] = features.high[i]
				self.low_test[i] = features.low[i]
			
		self.buy_and_hold_train = self.close_train[self.SIZE_TRAIN - 1] - self.close_train[0]
		self.buy_and_hold_test = self.close_test[self.SIZE_TEST - 1] - self.close_test[0]


	def load_prices(self,filename, mode):
		self.load_prices_cython(filename, mode)

	cdef run_simulation_cython(self, mode, actions, int max_pos):
		cdef:
			int i
			int pos = 0
			int prev_pos = 0
			int delta_pos = 0
			double delta_price = 0
			int deals = 0
			double price
			double res = 0.0
			double res_arr[2]

		if mode == 'train':
			for i in xrange(1,self.SIZE_TRAIN):
				
				delta_price = (self.low_train[i] + self.high_train[i])/2.0 - (self.low_train[i-1] + self.high_train[i-1])/2.0
				delta_pos = pos - prev_pos
				res += delta_pos*delta_price
				
				prev_pos = pos

				if actions[i] == 1:
					#buy
					if pos > max_pos:
						continue

					#res -= self.high_train[i]
					pos += 1
					deals += 1
				
				elif actions[i] == 2:
					#sell
					if pos < -max_pos:
						continue
					#res += self.low_train[i]
					pos -= 1
					deals += 1


			price = (self.close_train[self.SIZE_TRAIN - 1])
			'''if pos > 0:
				res -= pos*price;
			elif pos < 0:
				res += pos*price'''

			deals += abs(pos)

		elif mode == 'test':
			for i in xrange(1,self.SIZE_TEST):

				delta_price = (self.low_test[i] + self.high_test[i])/2.0 - (self.low_test[i-1] + self.high_test[i-1])/2.0
				delta_pos = pos - prev_pos
				res += delta_pos*delta_price
				
				prev_pos = pos

				if actions[i] == 1:
					#buy
					if pos > max_pos:
						self.results[i] = res
						continue

					#res -= self.high_train[i]
					pos += 1
					deals += 1

				elif actions[i] == 2:
					#sell
					if pos < -max_pos:
						self.results[i] = res
						continue

					#res += self.low_train[i]
					pos -= 1
					deals += 1

				self.results[i] = res

			price = (self.close_test[self.SIZE_TEST - 1])
			'''if pos > 0:
				res -= pos*price;
			elif pos < 0:
				res += pos*price'''

			deals += abs(pos)
			#self.results[0:i].tofile('results/results_' + filename + '_' + mode + '.txt')

		return [res,deals]

	def run_simulation(self, mode, actions, max_pos = 10):
		res_arr = self.run_simulation_cython(mode, actions, max_pos)
		return res_arr

	def get_buy_and_hold(self):
		return [self.buy_and_hold_train, self.buy_and_hold_test]