#####################################################
# Created by:   Dex Vo
# Date:         10/9/2018
# Class:        CSC 371 - Assignment 3
#####################################################
#   Given defined weights for both layers of the LVQ 
# network, this program will map a training neuron to its target
# class in the linear layer

from math import sqrt

# list all neurons
neurons = [[-1, 1, -1],
           [1, -1, -1],
           [-1, -1, 1],
           [1, -1, 1],
           [1, 1, -1],
           [-1, -1, -1],
           [-1, 1, 1]]

# define weights for first layer
weight1 = [[-1, 1, -1],
           [1, -1, -1],
           [-1, -1, 1],
           [1, -1, 1],
           [1, 1, -1],
           [-1, -1, -1],
           [-1, 1, 1]]

# define weights for second layer
weight2 = [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 1, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 0, 1]]

# find the Euclidean distance
def euclideanDistance(vect1, vect2):
	distance = 0.0
	for i in range(len(vect1)):
		distance += (vect1[i] - vect2[i])**2
	return sqrt(distance)

# find subclass by finding minimum Euclidean distance
def subClass(vect):
    euclidDist = euclideanDistance(vect, weight1[0])
    minDist = euclidDist
    pos = 0
    for i in range(len(weight1)):
        euclidDist = euclideanDistance(vect, weight1[i])
        if euclidDist < minDist:
            minDist = euclidDist
            pos = i
    a = []
    for i in range(len(weight1)):
        if i == pos:
            a.append(1)
        else:
            a.append(0)
    return a

# returns weight2 * a; where a is the subclass
def targetClass(vect):
    targetClass = [0, 0, 0]
    for i in range(len(weight2)):
        for j in range(len(weight2[i])):
            targetClass[j] += weight2[i][j] * vect[i]
    return targetClass


# Test network with each training neuron here:
a = subClass(neurons[0])
b = targetClass(a)
for i in range(len(b)):
    print(b[i])
