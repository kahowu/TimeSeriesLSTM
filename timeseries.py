from __future__ import print_function
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
import matplotlib.pyplot as plt

# Dummy data 
def create_data (): 
	data = [1] * 3 + [2] * 3
	data *= 3

	# Put this timeseries into a supervised dataset, where the target for each sample is the next sample
	ds = SequentialDataSet(1, 1)
	for sample, next_sample in zip(data, cycle(data[1:])):
	    ds.addSample(sample, next_sample)

	return ds 

def train (ds, net):
	# Train the network 
	trainer = RPropMinusTrainer(net, dataset=ds)
	train_errors = [] # save errors for plotting later
	EPOCHS_PER_CYCLE = 5
	CYCLES = 100
	EPOCHS = EPOCHS_PER_CYCLE * CYCLES
	for i in xrange(CYCLES):
	    trainer.trainEpochs(EPOCHS_PER_CYCLE)
	    train_errors.append(trainer.testOnData())
	    epoch = (i+1) * EPOCHS_PER_CYCLE
	    print("\r epoch {}/{}".format(epoch, EPOCHS))
	    stdout.flush()

	print()
	print("final error =", train_errors[-1])

	return train_errors, EPOCHS, EPOCHS_PER_CYCLE

def predict (ds, net):
	for sample, target in ds.getSequenceIterator(0):
	    print("               sample = %4.1f" % sample)
	    print("predicted next sample = %4.1f" % net.activate(sample))
	    print("   actual next sample = %4.1f" % target)
	    print()

def plot (train_errors, EPOCHS, EPOCHS_PER_CYCLE):
	plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
	plt.xlabel('epoch')
	plt.ylabel('error')
	plt.show()


if __name__ == "__main__":

	# Create dummy data
	ds = create_data()

	# Build a simple LSTM network with 1 input node, 5 LSTM cells and 1 output node
	net = buildNetwork(1, 5, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

	# Train the model
	train_errors, EPOCHS, EPOCHS_PER_CYCLE = train (ds, net)

	print("Done training. Plotting error")
	plot (train_errors, EPOCHS, EPOCHS_PER_CYCLE)

	# print("Predict next sample")
	# predict (ds, net)

