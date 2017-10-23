# forward and back-propagation for RNN within joint model of RNCRF
# only compute hidden state for each node in forward pass
# receive error from CRF to each node for back-propagation

import numpy as np
from util.math_util import *
import random

#define softmax function
def softmax(v):
    v = np.array(v)
    max_v = np.amax(v)
    e = np.exp(v - max_v)
    dist = e / np.sum(e)

    return dist

    
def der_tanh(x):
    return 1-np.tanh(x)**2

def forward_prop(params, tree, d, c, labels=True):

    tree.reset_finished()

    to_do = tree.get_nodes()

    (rel_dict, Wv, b, We) = params

    # forward prop
    while to_do:
        curr = to_do.pop(0)

        # node is leaf
        if len(curr.kids) == 0:

            # activation function is the tanh
            # compute hidden state
            curr.p = tanh(Wv.dot(curr.vec) + b)
            

        else:

            # - root isn't a part of this! 
            # - more specifically, the stanford dep. parser creates a superficial ROOT node
            #   associated with the word "root" that we don't want to consider during training
            # 'root' is the last one to be popped
            if len(to_do) == 0:
                # 'root' only has one kid, which is the root word
                ind, rel = curr.kids[0]
                curr.p = tree.get(ind).p
                
                continue
            
            # check if all kids are finished
            all_done = True
            for ind, rel in curr.kids:
                if tree.get(ind).finished == 0:
                    all_done = False
                    break

            # if not, push the node back onto the queue
            if not all_done:
                to_do.append(curr)
                continue

            # otherwise, compute p at node
            else:
                kid_sum = zeros( (d, 1) )
                for ind, rel in curr.kids:
                    curr_kid = tree.get(ind)

                    try:
                        kid_sum += rel_dict[rel].dot(curr_kid.p)

                    # - this shouldn't happen unless the parser spit out a seriously 
                    #   malformed tree
                    except KeyError:
                        print 'forward propagation error'
                        print tree.get_words()
                        print curr.word, rel, tree.get(ind).word
                
                kid_sum += Wv.dot(curr.vec)
                curr.p = tanh(kid_sum + b)


        # error and delta
        if labels:
            
            true_label = zeros( (c, 1) )
            for i in range(c):
                if curr.trueLabel == i:
                    true_label[i] = 1
                    
            curr.true_class = true_label
                    
            
        curr.finished = 1



# computes gradients for the given tree and increments existing gradients
def backprop(params, tree, d, c, len_voc, grads, mixed = False):

    (rel_dict, Wv, b) = params

    # start with root's immediate kid (for same reason as forward prop)
    ind, rel = tree.get(0).kids[0]
    root = tree.get(ind)

    # operate on tuples of the form (node, parent delta)
    to_do = [ (root, zeros( (d, 1) ) ) ]

    while to_do:
        curr = to_do.pop()
        node = curr[0]
        # parent delta
        delta_down = curr[1]
        delta = node.grad_h
        curr_der = der_tanh(node.p)
        node.delta_node = np.multiply(delta, curr_der)
        
        node.delta_full = delta_down + node.delta_node

        # internal node
        if len(node.kids) > 0:
            

            for ind, rel in node.kids:

                curr_kid = tree.get(ind)
                #W_rel
                grads[0][rel] += node.delta_full.dot(curr_kid.p.T)
                to_do.append( (curr_kid, rel_dict[rel].T.dot(node.delta_full) ) )


            grads[1] += node.delta_full.dot(node.vec.T)
            grads[2] += node.delta_full
            if mixed:
                grads[3][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[3][:, node.ind] += Wv.T.dot(node.delta_full).ravel()

        # leaf
        else:

            grads[1] += node.delta_full.dot(node.vec.T)
            grads[2] += node.delta_full

            if mixed:
                grads[3][50:, node.ind] += Wv.T.dot(node.delta_full).ravel()[50:]
            else:
                grads[3][:, node.ind] += Wv.T.dot(node.delta_full).ravel()
                
