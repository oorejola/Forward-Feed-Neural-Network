
"""
Fun Implemenation of Fully Connected Forward Feed Neural Network.
Using the Iris Data Set I will implemented a Feed Forward Neural Network as well as train using Back Propogation
and Stocastic Gradient Descent as our training rule. Built a very primetive Fully Connected FFNN class.

Author: Oliver Orejola


Parameters:
- Variable number of hidden layers and number of neurons per hidden layer
- Initialized weight matricies and biases can have varied interval of random numbers( e.g [-0.05,0.05] )
- Learning rate
- Number of Training Epochs




Authors notes:

Class representaiton was changed to a vector (e.g [1,0,0] = "Iris-setosa") to represent 
the 3 output nodes rather than use one output node and partition the range of a single 
node and interpret each interval as a certain class. (e.g if output in [0.0,0.333] 
interpret as "Iris-setosa"). It was a goal to minmize index chasing and preserve all
calculations in the form of matricies: often used in FeedForward and Back Propogation 
were matrix products, dot products, as well as outer products. Multiple functions were 
vectorized inorder to act linear component wise over matricies and vectors i.e numpy arrays
to limit non matrix operations. Each collection of weights connecting a pair of layers as well
as biases associated to each output unit were represented as a matrix. Utiliztion of matrix 
operations gave the advantage of varying the number of neurons in the hidden layer. 

"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from scipy.special import expit as sigmoid

#----------------------------------------------------------------------------------------------------------------#
#--------------------------------------- Defining Functions -----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
def class_representation_change(flower_type):
#Alter Flower Name to Binary Vector Representation.
	if flower_type == "Iris-setosa":
		return [1,0,0]
	if flower_type =="Iris-versicolor":
		return [0,1,0]
	else:
		return [0,0,1]

#Vectorize The Sigmoid Function inorder to act componentwize on vectors.
sig = np.vectorize(sigmoid)


class FFNN:

	def __init__(self,layer_info,input_size,output_size):
		self.layer = layer_info
		self.weights = self.Construct_Weights(layer_info,input_size,output_size)
		self.biases = self.Construct_Biases(layer_info,output_size)

	def Construct_Weights(self,layer_info,input_size,output_size):
		interval = [-0.05,0.1]
		np.random.seed(4)
		Weights = [np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[0],input_size))]
		for i in range(len(layer_info))[1:]:
			Weights.append(np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[i],layer_info[i-1])))
		n = len(layer_info)-1
		Weights.append(np.random.uniform(low=interval[0], high=interval[1], size=(output_size,layer_info[n])))
		return Weights

	def Construct_Biases(self,layer_info,output_size):
		interval = [-0.05,0.1]
		np.random.seed(4)
		Biases = [np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[0]))]
		for i in range(len(layer_info))[1:]:
			Biases.append(np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[i])))
		n = len(layer_info)-1
		Biases.append(np.random.uniform(low=interval[0], high=interval[1], size=(output_size)))
		return Biases
	# Computation of the Network
	# Takes input vector and feed forwards the information through network
	def FeedForward(self,x_in):
		outputs = [sig(self.weights[0].dot(x_in)+self.biases[0])]
		for i in range(len(self.weights))[1:]:
			outputs.append(sig(self.weights[i].dot(outputs[i-1])+self.biases[i]))
		return outputs
	
	def Update_Weights_Biases(self,new_weights,new_biases):
		self.weights = new_weights
		self.biases = new_biases
	


def Threshold(X):
	if X > .5:
		return 1
	else:
		return 0
Thresh = np.vectorize(Threshold)

# Back propogation training is based on Stochastic Graident Descent
# i.e The weights and biases are updated based on the data recieved and error calculated
# for a particular instance of data.

def BackPropogation(x_in, NN ,t,R):
	# t = Target Output, R = Learning Rate
	outputs = NN.FeedForward(x_in)
	n = len(outputs)
	error_terms = [outputs[n-1]*(1-outputs[n-1])*(t-outputs[n-1])]
	
	for i in range(n)[1:]:
		error_terms.insert(0, outputs[n-1-i]*(1-outputs[n-1-i])*(NN.weights[n-i].T).dot(error_terms[len(error_terms)-i]))
	
	new_weights = [NN.weights[i] + R * np.outer(error_terms[i],outputs[i-1]) for i in range(n)[1:]]
	new_weights.insert(0,NN.weights[0]+ R * np.outer(error_terms[0],x_in))
	new_biases = [NN.biases[i] + R * error_terms[i] for i in range(n)]

	return new_weights, new_biases




#----------------------------------------------------------------------------------------------------------------#

def main():

#-------------------------------#
#-- Import Data Using Pandas. --#
#-------------------------------#

	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	irisdata = pd.read_csv("iris.csv",header = None, names = names)

#splitting dataset to validation and train
	X = irisdata.values[:,:4] 	#Input Features
	Y = irisdata.values[:,4]	#Classes

#Change Classes to binary vectors representing each Iris Class
	Y = np.array([ class_representation_change(x) for x in Y])


#use train_test_split from mode_selection to randomize ordered data and split data to test on 30% Test and 70% Train of the data set
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.30, random_state=7)
#-------------------------------#


#-----------------------------------#
#------------ Parameters -----------#
#-----------------------------------#
	Layers = [2,3]
	In_size = 4
	Out_size = 3
	Rate = .05
	Num_Epochs = 1000

# Define Forward Feed Neural Network

	NN = FFNN(Layers,In_size,Out_size)

#----------------------------------#
#-------------- Train  ------------#
#----------------------------------#

	print "\nBegin Training.\n"
	for j in range(Num_Epochs):
		for i in range(len(X_train)):
			New_Weights, New_Biases = BackPropogation(X_train[i],NN,Y_train[i],Rate)
			NN.Update_Weights_Biases(New_Weights,New_Biases)
		if (j+1)%100 == 0:
			print "Training on Epoch: %d"% (j+1)
	print "\n"*2

#----------------------------------#
#--- Test and Train Accuracy  -----#
#----------------------------------#

	Test_Accuracy = 0.0
	for i in range(len(X_validation)):
		if np.array_equal(Thresh(NN.FeedForward(X_validation[i])[-1:][0]), Y_validation[i]):
			Test_Accuracy+=1
	Test_Accuracy = Test_Accuracy/len(X_validation)
	print "Accuracy on test data %f" % Test_Accuracy
	print"-"*30



#----------------------------------#
#-- Forward Feed Network Details --#
#----------------------------------#
	print "\n"
	print "Network Details: "
	print"-"*30
	print "Input Size: %d" % In_size
	print "Output Classifier Size: %d" % Out_size
	print "Number of Hidden Layers: %d" %len(Layers)
	for i in range(len(Layers)):
		print "  Hidden Layer %d: %d neurons" % (i+1 , Layers[i])

#----------------------------------#
#---- Network Training Details ----#
#----------------------------------#
	print "\n"
	print "Network Training Details: "
	print "-"*30
	print "Number of Data Points Used to Train : %d" % len(X_train)
	print "Learning rate: %f" % Rate
	print "Number of Epochs: %d" % Num_Epochs
	print "\n"

if __name__== "__main__":
  main()
