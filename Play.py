import numpy as np
import pandas as pd
from sklearn import model_selection
from scipy.special import expit as sigmoid



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

#Computation of the Network
#Takes input vector and returns expected output
def FeedForward(x_in, weights,biases):
	outputs = [sig(weights[0].dot(x_in)+biases[0])]
	for i in range(len(weights))[1:]:
		outputs.append(sig(weights[i].dot(outputs[i-1])+biases[i]))
	return outputs

def Threshold(X):
	if X > .5:
		return 1
	else:
		return 0
Thresh = np.vectorize(Threshold)

def BackPropogation(x_in, weights,biases,t,R):
	# t = Target Output, R = Learning Rate
	outputs = FeedForward(x_in,weights,biases)
	n = len(outputs)
	print outputs[n-1]
	error_terms = [outputs[n-1]*(1-outputs[n-1])*(t-outputs[n-1])]
	
	for i in range(n)[1:]:
		error_terms.insert(0, outputs[n-1-i]*(1-outputs[n-1-i])*(weights[n-i].T).dot(error_terms[len(error_terms)-i]))
	
	new_weights = [weights[i] + R * np.outer(error_terms[i],outputs[i-1]) for i in range(n)[1:]]
	new_weights.insert(0,weights[0]+ R * np.outer(error_terms[0],x_in))
	new_biases = [biases[i] + R * error_terms[i] for i in range(n)]

	return new_weights, new_biases

def Construct_Weights(layer_info,input_size,output_size):
	interval = [-0.05,0.1]
	np.random.seed(4)
	Weights = [np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[0],input_size))]
	for i in range(len(layer_info))[1:]:
		Weights.append(np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[i],layer_info[i-1])))
	n = len(layer_info)-1
	Weights.append(np.random.uniform(low=interval[0], high=interval[1], size=(output_size,layer_info[n])))
	return Weights

def Construct_Biases(layer_info,output_size):
	interval = [-0.05,0.1]
	np.random.seed(4)
	Biases = [np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[0]))]
	for i in range(len(layer_info))[1:]:
		Biases.append(np.random.uniform(low=interval[0], high=interval[1], size=(layer_info[i])))
	n = len(layer_info)-1
	Biases.append(np.random.uniform(low=interval[0], high=interval[1], size=(output_size)))
	return Biases



Layers = [4,2,3]
In_size = 4
Out_size = 3
Rate = 0.05
Num_Epochs = 1000


print Layers[-1:][0]
yaya= Construct_Weights(Layers,In_size,Out_size)
for x in yaya:
	print x.shape
Biases = Construct_Biases(Layers,Out_size)
for x in Biases:
	print x.shape
print "\n"
results = FeedForward([1,0,.4,2],yaya,Biases)
for x in results:
	print x.shape

Weights, Biases = BackPropogation([1,0,.4,2],yaya,Biases,[1,0,0],Rate)
for x in Weights:
	print x.shape

for x in Biases:
	print x.shape

#ForwardFeedNetworkDetails
print "\n"
print "Network Details: "
print"-"*30
print "Input Size: %d" % In_size
print "Output Classifier Size: %d" % Out_size
print "Number of Hidden Layers: %d" %len(Layers)
for i in range(len(Layers)):
	print "  Hidden Layer %d: %d neurons" % (i+1 , Layers[i])
print "\n"
print "Learning Rate: %f" %Rate



