from __future__ import print_function
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import sklearn as sk 
import pickle

close_max = 0
open_max = 0

def train (ds, net):
	# Train the network 
	trainer = RPropMinusTrainer(net, dataset=ds)
	train_errors = [] # save errors for plotting later
	EPOCHS_PER_CYCLE = 5
	CYCLES = 100
	EPOCHS = EPOCHS_PER_CYCLE * CYCLES
	for i in xrange(CYCLES):
	    trainer.trainEpochs(EPOCHS_PER_CYCLE)
	    error = trainer.testOnData()
	    train_errors.append(error)
	    epoch = (i+1) * EPOCHS_PER_CYCLE
	    print("\r epoch {}/{}".format(epoch, EPOCHS))
	    stdout.flush()

	# print("final error =", train_errors[-1])

	return train_errors, EPOCHS, EPOCHS_PER_CYCLE

def predict (ds, net, date):
	i = 0 
	filename = "Result.csv"
	f = open(filename, 'w')
	f.write('Date, Opening price, Predicted closing price, Actual closing price \n')
	for sample, target in ds.getSequenceIterator(0):
		s = '{0},{1},{2}, {3}\n'.format(date[i], (sample * open_max), (net.activate(sample)* close_max), (target * close_max))
		print (s)
		f.write(s)
		i += 1
	print ("Created " + filename)
	f.close()

def plot (train_errors, EPOCHS, EPOCHS_PER_CYCLE):
	plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
	plt.xlabel('epoch')
	plt.ylabel('error')
	plt.show()

def create_train_set (open_price, close_price):
	global open_max
	global close_max 
	ds = SequentialDataSet(1, 1)
	open_data = normalize (open_price) 
	close_data = normalize (close_price) 
	open_max = open_data[1]
	close_max = close_data[1]
	open_price = open_data[0]
	close_price = close_data[0]

	size = len (open_price)
	for i in range(0, size):
		ds.addSample(open_price[i], close_price[i])	

	return ds 

def calculate_mse (ds, net):
	diff_list = []
	for sample, target in ds.getSequenceIterator(0):
		squared_diff = np.square((net.activate(sample) * close_max - target * close_max))
		diff_list.append(squared_diff)

	mse = float(sum (diff_list)) / float(len (diff_list))

	print ("The mean squared error is ", mse)
	return mse

def normalize (data):
	maxnum = max (data)
	normData = data / maxnum
	return (normData, maxnum)


def save_model (net):
	filename = "model.pkl"
	fileObject = open(filename, 'w')
	pickle.dump(net, fileObject)
	fileObject.close()

if __name__ == "__main__":
	# Load data 
	arg = sys.argv
	filename = arg[1]
	filedir = "./data/" + filename
	df = pd.read_csv(filedir)
	open_price = df["Open"]
	close_price = df["Close"]
	date = df["Date"]
	ds = create_train_set (open_price, close_price)

	print("Building network...")
	# Build a simple LSTM network with 1 input node, 5 LSTM cells and 1 output node
	net = buildNetwork(1, 5, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

	# Train the model
	print("Start training...")
	train_errors, EPOCHS, EPOCHS_PER_CYCLE = train (ds, net)

	print("Done training. Plotting error...")
	plot (train_errors, EPOCHS, EPOCHS_PER_CYCLE)

	print("Predicting price...")
	predict(ds, net, date)

	print("Calculating mean squared error...")
	calculate_mse (ds, net)

	print("Saving model...")
	save_model(net)



