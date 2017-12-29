import numpy as np

def main():
	#I'm sure there's a more efficient way to store the children of a node
	child3 = [2, 28] #Creates children of nodes, node labels correspond to depth of node, similar to how they would be stored in a priority queue.
	child4 = [8, 2]
	child5 = [22, 3]
	child6 = [8, 7]
	child1 = [10, 30, child3, child4]
	child2 = [30, 10, child5, child6]
	root = [40, 40, child1, child2]
	#Prints infogain from 3 nodes on tree
	print("infogain from first node: ", infogain(root, child1, child2))
	print("infogain from second node: ", infogain(child1, child3, child4))
	print("infogain from third node: ", infogain(child2, child5, child6))

def prob(node):
	# Returns probability of a given node
	return node[0]/(node[0]+node[1])

def entropy(node):
	# Returns entropy of a node by using Logarithm and the probability functions from earlier
	return (-prob(node)*np.log2(prob(node)))-((1-prob(node))*np.log2(1-prob(node)))
def infogain(parent, left, right):
	#Returns infogain from three nodes, using entropy commands from before
	return entropy(parent)-(prob(left)*entropy(left))-(prob(right)*entropy(right))
main()
#Calls Main Function
