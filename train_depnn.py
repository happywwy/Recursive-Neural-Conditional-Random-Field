# this file pre-trains a recursive neural network with dependency trees
# without any CRF on top

import numpy as np
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from rnn.adagrad import Adagrad
import rnn.propagation as prop
import cPickle, time, argparse
import random

#computes parameter updates with regularization
def par_objective(data, rel_dict, Wv, Wc, b, b_c, L, d, c, len_voc, rel_list, lambdas):

    #non-data params
    params = (rel_dict, Wv, Wc, b, b_c, L)
    oparams = [params, d, c, len_voc, rel_list]

    param_data = []
    param_data.append(oparams)
    param_data.append(data)
    
    #gradient and error  
    result = objective_and_grad(param_data)
    [total_err, grads, all_nodes] = result

    # add L2 regularization
    [lambda_W, lambda_L, lambda_C] = lambdas

    reg_cost = 0.0
    #regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * sum(rel_dict[key] ** 2)
        grads[0][key] = grads[0][key] / all_nodes
        grads[0][key] += lambda_W * rel_dict[key]
    
    #regularization for transformation matrix Wv
    reg_cost += 0.5 * lambda_W * sum(Wv ** 2)
    grads[1] = grads[1] / all_nodes
    grads[1] += lambda_W * Wv
    
    #regularization for classification matrix Wc
    reg_cost += 0.5 * lambda_C * sum(Wc ** 2)
    grads[2] = grads[2] / all_nodes
    grads[2] += lambda_C * Wc

    #regularization for bias b
    grads[3] = grads[3] / all_nodes
    
    #regularization for bias b_c
    grads[4] = grads[4] / all_nodes

    reg_cost += 0.5 * lambda_L * sum(L ** 2)
    
    #regularization for word embedding matrix
    grads[5] = grads[5] / all_nodes
    grads[5] += lambda_L * L

    cost = total_err / all_nodes + reg_cost

    return cost, grads


# this function computes the objective / grad for each minibatch
def objective_and_grad(par_data):

    params, d, c, len_voc, rel_list = par_data[0]
    data = par_data[1]
    
    # returns list of initialized zero gradients which backprop modifies
    grads = init_dtrnn_grads(rel_list, d, c, len_voc)
    (rel_dict, Wv, Wc, b, b_c, L) = params

    error_sum = 0.0
    num_nodes = 0
    tree_size = 0

    # compute error and gradient for each tree in minibatch
    # also keep track of total number of nodes in minibatch
    for index, tree in enumerate(data):

        nodes = tree.get_nodes()
        for node in nodes:
            node.vec = L[:, node.ind].reshape( (d, 1) )

        prop.forward_prop(params, tree, d, c)
        error_sum += tree.error()
        tree_size += len(nodes)

        prop.backprop(params[:-1], tree, d, c, len_voc, grads)

    return (error_sum, grads, tree_size)
    

# train save model
if __name__ == '__main__':
    
    seed_i = 12

    # command line arguments
    parser = argparse.ArgumentParser(description='DT-RNN: a dependency tree-based recursive neural network')
    parser.add_argument('-data', help='location of dataset', default='util/data_semEval/final_input_sample')
    parser.add_argument('-We', help='location of word embeddings', default='util/data_semEval/word_embeddings_sample')
    parser.add_argument('-We_mixed', help='location of word embeddings mixed', default='util/data_semEval/word_embeddings_mixed')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=100)
    
    # no of classes
    parser.add_argument('-c', help='number of classes', type=int, default=5)
    parser.add_argument('-lW', '--lambda_W', help='regularization weight for composition matrices', \
                        type=float, default=0.001)
    parser.add_argument('-lWe', '--lambda_We', help='regularization weight for word embeddings', \
                        type=float, default=0.001)
    # regularization for classification matrix
    parser.add_argument('-lWc', '--lambda_Wc', help='regularization weight for classification matrix', \
                        type=float, default=0.001)                    
                    
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size', type=int, default=25)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=1)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=30)
    parser.add_argument('-v', '--do_val', help='check performance on dev set after this many\
                             epochs', type=int, default=4)
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='models/rnn_params_sample')
                         
    parser.add_argument('-op', help='use mixed word vector or not', default = False)
    parser.add_argument('-len', help='training vector length for mixed', default = 50)

    args = vars(parser.parse_args())
    

    ## load data
    vocab, rel_list, tree_dict = \
        cPickle.load(open(args['data'], 'rb'))

    train_trees, test_trees = tree_dict[:75], tree_dict[75:]
    #val_trees = tree_dict['dev']
	
    # choice of mixed word embedding (some can be updated, some are fixed)
    if args['op']:
        orig_We = cPickle.load(open(args['We_mixed'], 'rb'))
    else:
        orig_We = cPickle.load(open(args['We'], 'rb'))

    # regularization lambdas
    lambdas = [args['lambda_W'], args['lambda_We'], args['lambda_Wc']]

    # output log and parameter file destinations
    param_file = args['output']
    # "training_log"
    log_file = param_file.split('_')[0] + '_log'

    print 'number of training sentences:', len(train_trees)
    rel_list.remove('root')
    print 'number of dependency relations:', len(rel_list)
    # number of classes
    print 'number of classes:', args['c']

    ## remove incorrectly parsed sentences from data
    # print 'removing bad trees train...'
    bad_trees = []
    for ind, tree in enumerate(train_trees):
        
        #add condition when the tree is empty
        if tree.get_nodes() == []:
            bad_trees.append(ind)

        elif tree.get(0).is_word == 0:
            print tree.get_words(), ind
            bad_trees.append(ind)

    # pop bad trees, higher indices first
    # print 'removed ', len(bad_trees)
    for ind in bad_trees[::-1]:
        #train_trees.pop(ind)
        train_trees = np.delete(train_trees, ind)

    # generate params / We
    # d = word embedding dimension
    # Returns (dict{rels:[mat]}, Wv, Wc, b, b_c)
    if (args['op']):
        params = gen_dtrnn_params(args['d'] + args['len'], args['c'], rel_list)
    else:
        params = gen_dtrnn_params(args['d'], args['c'], rel_list)
    rel_list = params[0].keys()

    # add We matrix to params
    params += (orig_We, )
    # r is 1-D param vector
    r = roll_params(params, rel_list)

    dim = r.shape[0]
    print 'parameter vector dimensionality:', dim

    log = open(log_file, 'w')
    paramfile = open( param_file, 'wb')

    # minibatch adagrad training
    ag = Adagrad(r.shape)
    (rel_dict, Wv, Wc, b, b_c, We) = params

    for tdata in [train_trees]:

        min_error = float('inf')

        for epoch in range(0, args['num_epochs']):

            lstring = ''
            
            random.seed(seed_i)
            # create mini-batches
            random.shuffle(tdata)
            batches = [tdata[x : x + args['batch_size']] for x in xrange(0, len(tdata), 
                       args['batch_size'])]

            epoch_error = 0.0
            for batch_ind, batch in enumerate(batches):
                now = time.time()

                # return cost, grad  
                if args['op']:
                    err, grads = par_objective(batch, rel_dict, Wv, Wc, b, b_c, We, args['d'] + args['len'], \
                                               args['c'], len(vocab), rel_list, lambdas)
                else:
                    err, grads = par_objective(batch, rel_dict, Wv, Wc, b, b_c, We, args['d'], \
                                               args['c'], len(vocab), rel_list, lambdas)
                                          
                grad = roll_params(grads, rel_list)                          
                update = ag.rescale_update(grad)
                updates = unroll_params(update, args['d'], args['c'], len(vocab), rel_list)
                for rel in rel_list:
                    rel_dict[rel] -= updates[0][rel]
                Wv -= updates[1]
                Wc -= updates[2]
                b -= updates[3]
                b_c -= updates[4]
                We -= updates[5]

                lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                        ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
                print lstring
                log.write(lstring + '\n')
                log.flush()

                epoch_error += err

            # done with epoch
            print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
            lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                     + ' min error = ' + str(min_error) + '\n\n'
            log.write(lstring)
            log.flush()

            # save parameters if the current model is better than previous best model
            if epoch_error < min_error:
                min_error = epoch_error
                print 'saving model...'
                params = (rel_dict, Wv, Wc, b, b_c, We)

            # reset adagrad weights
            if epoch % args['adagrad_reset'] == 0 and epoch != 0:
                ag.reset_weights()

            # check accuracy on validation set
            """
            if epoch % args['do_val'] == 0 and epoch != 0:
                print 'validating...'
                params = unroll_params(r, args['d'], args['c'], len(vocab), rel_list)
                train_acc, val_acc = validate([train_trees, val_trees], params, args['d'])
                lstring = 'train acc = ' + str(train_acc) + ', val acc = ' + str(val_acc) + '\n\n\n'
                print lstring
                log.write(lstring)
                log.flush()
            
            """
    cPickle.dump( ( params, vocab, rel_list), paramfile)
    
    log.close()
    paramfile.close()
    
    

