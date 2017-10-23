# -*- coding: utf-8 -*-
import numpy as np
from util.gen_util import *
from util.math_util import *
from util.dtree_util import *
from rnn.adagrad_crf import Adagrad
import rnn.crf_propagation as prop
#from classify.learn_classifiers import validate
import cPickle, time, argparse

import pycrfsuite

#for ordered dictionary
from collections import OrderedDict
import gc
import os, sys


def evaluate(epoch, inst_ind, data, rel_dict, Wv, b, We, vocab, rel_list, d, c, mixed = False):
    #output labels
    tagger = pycrfsuite.Tagger()
    tagger.open(str(epoch)+str(inst_ind)+'crf.model')
    
    #word2vec
    dic_file = open('util/data_semEval/w2v_sample.txt', 'r')
    dic = dic_file.readlines()

    dictionary = {}

    for line in dic:
        word_vector = line.split(",")
        word = ','.join(word_vector[:len(word_vector) - d - 1])
    
        vector_list = []
        for element in word_vector[len(word_vector) - d - 1:len(word_vector) - 1]:
            vector_list.append(float(element))
        
        vector = np.asarray(vector_list)
        dictionary[word] = vector
    

        
    test_trees = data
    
    bad_trees = []
    for ind, tree in enumerate(test_trees):
        if tree.get(0).is_word == 0:
            # print tree.get_words()
            bad_trees.append(ind)
            continue

    # print 'removed', len(bad_trees)
    for ind in bad_trees[::-1]:
        test_trees = np.delete(test_trees, ind)
      
    true = []
    predict = []  
    
    
    count = 0
    
    for ind, tree in enumerate(test_trees):
        nodes = tree.get_nodes()
        sent = []
        h_input = np.zeros((len(tree.nodes) - 1, d))
        y_label = np.zeros((len(tree.nodes) - 1,), dtype = int)
       
        
        for index, node in enumerate(nodes):
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape( (d, 1) )
            elif node.word.lower() in dictionary.keys():
                if mixed:
                    node.vec = (dictionary[node.word.lower()].append(2 * np.random.rand(50) - 1)).reshape( (d, 1) )
                else:
                    node.vec = dictionary[node.word.lower()].reshape(d, 1)
            else:
                node.vec = np.random.rand(d,1)
                count += 1
            
        prop.forward_prop([rel_dict, Wv, b, We], tree, d, c, labels=False)
        
        for index, node in enumerate(tree.nodes): 
            
            if index != 0:
                
                if tree.get(index).is_word == 0:
                    y_label[index - 1] = 0
                    sent.append(None)
                    
                    for i in range(d):
                        h_input[index - 1][i] = 0
                else:
                    y_label[index - 1] = node.trueLabel
                    sent.append(node.word)
                    
                    for i in range(d):
                        h_input[index - 1][i] = node.p[i]

        crf_sent_features = sent2features(d, sent, h_input)
        for item in y_label:
            true.append(str(item))
        
        #predict           
        prediction = tagger.tag(crf_sent_features)
        for label in prediction:
            predict.append(label)
        
        



#convert pos tag to one-hot vector
def pos2vec(pos):
    
    pos_list = ['PUNCT', 'SYM', 'CONJ', 'NUM', 'DET', 'ADV', 'X', 'ADP', 'ADJ', 'VERB', \
    'NOUN', 'PROPN', 'PART', 'PRON', 'INTJ']
            
    ind = pos_list.index(pos)
    vec = np.zeros(15)
    vec[ind] = 1
            
    return vec
    
#convert word to its namelist feature
def name2vec(sent, i, name_term, name_word):
    
    word = sent[i]
    name_vec = [0., 0.]
    
    if word != None:
        for term in name_term:
            if word == term:
                name_vec[0] = 1.
            elif i == 0 and len(sent) > 1 and sent[i + 1] != None:
                if word + ' ' + sent[i + 1] in term:
                    name_vec[0] = 1.
            elif i == len(sent) - 1 and len(sent) > 1 and sent[i - 1] != None:
                if sent[i - 1] + ' ' + word in term:
                    name_vec[0] = 1.
            elif i > 0 and i < len(sent) - 1:
                if (sent[i + 1] != None and word +' '+ sent[i + 1] in term) \
                    or (sent[i - 1] != None and sent[i - 1] +' '+ word in term):
                    
                    name_vec[0] = 1.
                
        if word in name_word:
            name_vec[1] = 1.
        
    return name_vec


# for constructing crf word features by taking rnn hidden representation
def word2features(d, sent, h_input, i):
    
    #for ordered dictionary
    word_features = OrderedDict()    
    
    word_features['bias'] = 1.

    #if it is punctuation
    if sent[i] == None:
        word_features['punkt'] = 1.
    
    else:    
        for n in range(d):
            word_features['worde=%d' % n] = h_input[i,n]
    ''' 
    # you can append human-engineered features here by providing corresponding lists
    #add pos features
    for n in range(15):
        word_features['pos=%d' % n] = pos_mat[i, n]
  
    
    #add namelist features
    name_vec = name2vec(sent, i, name_term, name_word)
    word_features['namelist1'] = name_vec[0]
    word_features['namelist2'] = name_vec[1]
    
    # add opinion word lexicon
    if sent[i] in senti_list:
        word_features['sentiment'] = 1.
    else:
        word_features['sentiment'] = 0.
    '''
    if i > 0 and sent[i - 1] == None:
        word_features['-1punkt'] = 1.  
        
    elif i > 0:
        for n in range(d):
            word_features['-1worde=%d' %n] = h_input[i - 1, n]
        '''
        #add pos features
        for n in range(15):
            word_features['-1pos=%d' % n] = pos_mat[i - 1, n]
        
        #add namelist features
        name_vec = name2vec(sent, i - 1, name_term, name_word)
        word_features['-1namelist1'] = name_vec[0]
        word_features['-1namelist2'] = name_vec[1]
        
        # add opinion word lexicon
        if sent[i-1] in senti_list:
            word_features['-1sentiment'] = 1.
        else:
            word_features['-1sentiment'] = 0.
        '''
        
    else:
        word_features['BOS'] = 1.

    if i < len(sent) - 1 and sent[i + 1] == None:
        word_features['+1punkt'] = 1.
        
    elif i < len(sent) - 1:
        for n in range(d):
            word_features['+1worde=%d' %n] = h_input[i + 1, n]
        '''
        #add pos features
        for n in range(15):
            word_features['+1pos=%d' % n] = pos_mat[i + 1, n]
       
        #add namelist features
        name_vec = name2vec(sent, i + 1, name_term, name_word)
        word_features['+1namelist1'] = name_vec[0]
        word_features['+1namelist2'] = name_vec[1]
        
        # add opinion word lexicon
        if sent[i+1] in senti_list:
            word_features['+1sentiment'] = 1.
        else:
            word_features['+1sentiment'] = 0.
        '''
        
    else:
        word_features['EOS'] = 1.
        
    return word_features

# for constructing crf feature mapping for a sentence
def sent2features(d, sent, h_input):
    return pycrfsuite.ItemSequence([word2features(d, sent, h_input, i) for i in range(len(sent))])

'''
from memory_profiler import profile
@profile
'''

# compute gradients and updates
# boolean determines whether to save the crf model
def par_objective(epoch, data, rel_dict, Wv, b, L, Wcrf, d, c, len_voc, \
    rel_list, lambdas, trainer, num, eta, dec, boolean):
    
    #initialize gradients
    grads = init_crfrnn_grads(rel_list, d, c, len_voc)

    error_sum = np.zeros(1)
    num_nodes = 0
    tree_size = 0
    
    #compute for one instance
    tree = data
    nodes = tree.get_nodes()
    
    for node in nodes:
        node.vec = L[:, node.ind].reshape( (d, 1) )
    
    prop.forward_prop([rel_dict, Wv, b, L], tree, d, c)
    tree_size += len(nodes)
    
    #after a rnn forward pass, compute crf
    sent = []
    #input matrix composed of hidden vector from RNN
    h_input = np.zeros((len(tree.nodes) - 1, d))
    y_label = np.zeros((len(tree.nodes) - 1,), dtype = int)
   
    for ind, node in enumerate(tree.nodes):
        if ind != 0:
            
            #if current token is punctuation
            if tree.get(ind).is_word == 0:
                y_label[ind - 1] = 0
                sent.append(None)
                
                for i in range(d):
                    h_input[ind - 1][i] = 0
            #if current token is a word
            else:
                y_label[ind - 1] = node.trueLabel
                sent.append(node.word)
                
                for i in range(d):
                    h_input[ind - 1][i] = node.p[i]

    crf_sent_features = sent2features(d, sent, h_input) 
    #when parameters are updated, hidden vectors are also updated for crf input
    #this is for updating CRF input features, num is the index of the instance
    trainer.modify(crf_sent_features, num)
    # crf feature dimension
    attr_size = 3 * (d + 1) + 3
    d_size = (len(tree.nodes) - 1) * attr_size
    #delta for hidden matrix from crf
    delta_features = np.zeros(d_size)
    
    #check if we need to store the model
    if boolean == True:
        trainer.train(model=str(epoch)+str(num)+'crf.model', weight=Wcrf, delta=delta_features, inst=num, eta=eta, decay=dec, loss=error_sum, check=1)
    else:
        trainer.train(model='', weight=Wcrf, delta=delta_features, inst=num, eta=eta, decay=dec, loss=error_sum, check=1)

    grad_h = []
    start = 0
    #pass delta h to separate feature vectors to backpropagate to rnn
    for ind, node in enumerate(tree.nodes):
        if ind != 0:
            grad_h.append(-delta_features[start: start + attr_size])
            start += attr_size
    
    for ind, node in enumerate(tree.nodes):
        if ind != 0:
            if tree.get(ind).is_word != 0:
                node.grad_h = grad_h[ind - 1][1: d + 1].reshape(d, 1)
                #check if the sentence only contains one word
                if len(tree.nodes) > 2:
                    if ind == 1:
                        if tree.get(ind + 1).is_word != 0:
                            node.grad_h += grad_h[ind][2 * d + 2: 3 * d + 2].reshape(d, 1)
                            
                    elif ind < len(sent) - 1:
                        if tree.get(ind + 1).is_word != 0:
                            node.grad_h += grad_h[ind][2 * d + 2: 3 * d + 2].reshape(d, 1)
                        if tree.get(ind - 1).is_word != 0:
                            node.grad_h += grad_h[ind - 2][d + 2: 2 * d + 2].reshape(d, 1)
                    else:
                        if tree.get(ind - 1).is_word != 0:
                            node.grad_h += grad_h[ind - 2][d + 2: 2 * d + 2].reshape(d, 1)

    
    prop.backprop([rel_dict, Wv, b], tree, d, c, len_voc, grads)
    [lambda_W, lambda_L] = lambdas
   
    reg_cost = 0.0
    #regularization for relation matrices
    for key in rel_list:
        reg_cost += 0.5 * lambda_W * sum(rel_dict[key] ** 2)
        grads[0][key] = grads[0][key] / tree_size
        grads[0][key] += lambda_W * rel_dict[key]
    #regularization for transformation matrix and bias
    reg_cost += 0.5 * lambda_W * sum(Wv ** 2)
    grads[1] = grads[1] / tree_size
    grads[1] += lambda_W * Wv
    grads[2] = grads[2] / tree_size
    #regularization for word embedding
    reg_cost += 0.5 * lambda_L * sum(L ** 2)
    grads[3] = grads[3] / tree_size
    grads[3] += lambda_L * L

    cost = error_sum[0] + reg_cost

    return cost, grads, Wcrf


#create new function for initializating trainer for CRF (feature map)
def trainer_initialization(m_trainer, trees, params, d, c, len_voc, rel_list):
    param_list = unroll_params_noWcrf(params, d, c, len_voc, rel_list)
    (rel_dict, Wv, b, L) = param_list

    
    for tree in trees:
        nodes = tree.get_nodes()
    
        for node in nodes:
            node.vec = L[:, node.ind].reshape( (d, 1) )
        
        prop.forward_prop(param_list, tree, d, c)
        
        sent = []
        #input feature matrix to crf for the sentence 
        h_input = np.ones((len(tree.nodes) - 1, d))
        y_label = np.zeros((len(tree.nodes) - 1,), dtype = int)
        
        
        for ind, node in enumerate(tree.nodes):
            if ind != 0:
                
                if tree.get(ind).is_word == 0:
                    y_label[ind - 1] = 0
                    sent.append(None)
                    
                    for i in range(d):
                        h_input[ind - 1][i] = 0
                else:
                    y_label[ind - 1] = node.trueLabel
                    sent.append(node.word)
                    
                    for i in range(d):
                        h_input[ind - 1][i] = node.p[i]
                
        y_label = np.asarray(y_label)

        crf_sent_features = sent2features(d, sent, h_input)
        crf_sent_labels = [str(item) for item in y_label] 
        m_trainer.append(crf_sent_features, crf_sent_labels)
        
    return m_trainer


# train and save model
if __name__ == '__main__':


    # command line arguments
    parser = argparse.ArgumentParser(description='RNCRF: a joint model for aspect-based sentiment analysis')
    parser.add_argument('-data', help='location of dataset', default='util/data_semEval/final_input_sample')
    # add model parameter (pretrained from DT-RNN)
    parser.add_argument('-pretrain_model', help='location of pretrained model', default='models/rnn_params_sample')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=100)
    
    # no of classes
    parser.add_argument('-c', help='number of classes', type=int, default=5)

    parser.add_argument('-lW', '--lambda_W', help='regularization weight for composition matrices', \
                        type=float, default=0.0001)
    parser.add_argument('-lWe', '--lambda_We', help='regularization weight for word embeddings', \
                        type=float, default=0.0001)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=1)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=50)
    """
    parser.add_argument('-v', '--do_val', help='check performance on dev set after this many\
                         epochs', type=int, default=4)
    """
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='final_model/final_params_sample')
                         
    parser.add_argument('-op', help='use mixed word vector or not', default = False)
    parser.add_argument('-len', help='training vector length', default = 50)

    args = vars(parser.parse_args())
        
    ## load data
    vocab, rel_list, tree_dict = \
        cPickle.load(open(args['data'], 'rb'))

    train_trees, test_trees = tree_dict[:75], tree_dict[75:]

    #import pre-trained model parameters
    params, vocab, rel_list = cPickle.load(open(args['pretrain_model'], 'rb'))
    (rel_dict, Wv, Wc, b, b_c, We) = params

    # regularization lambdas
    lambdas = [args['lambda_W'], args['lambda_We']]

    # output log and parameter file destinations
    param_file = args['output']
    # "training_log"
    log_file = param_file.split('_')[0] + '_log'

    print 'number of training sentences:', len(train_trees)
    #print 'number of validation sentences:', len(val_trees)
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
    
    #add train_size
    train_size = len(train_trees)

    c = args['c']
    d = args['d']

    #crf weight matrix
    Wcrf = np.zeros(c * (3 * (d + 1 + 1)) + 18) #dimension
    # store delta for CRF backpropagation
    delta_features = np.zeros(3000)

    # r is 1-D param vector
    r_noWcrf = roll_params_noWcrf((rel_dict, Wv, b, We), rel_list)

    dim = r_noWcrf.shape[0]
    print 'parameter vector dimensionality:', dim

    log = open(log_file, 'w')
    crf_loss = np.zeros(1)

    # minibatch adagrad training
    ag = Adagrad(r_noWcrf.shape)
    
    #initialize trainer object for CRF
    trainer = pycrfsuite.Trainer(algorithm='l2sgd', verbose=False)
    trainer.set_params({
    'c2': 1.,
    'max_iterations': 1  # stop earlier
    })
    
    trainer = trainer_initialization(trainer, train_trees, r_noWcrf, d, c, len(vocab), rel_list)
    trainer.train(model='', weight=Wcrf, delta=delta_features, inst=0, eta=0, decay=0, loss=crf_loss, check=0)    
    
    #learning rate decay
    t=-1.
    lamb = 2. * 1. / train_size
    t_0 = 1. / (lamb * 0.02)
    
    for tdata in [train_trees]:

        min_error = float('inf')

        for epoch in range(0, args['num_epochs']):
            
            #param_file = args['output'] + str(epoch)
            #paramfile = open( param_file, 'wb')
            
            decay = 1.
            lstring = ''

            epoch_error = 0.0
            
            for inst_ind, inst in enumerate(tdata):
                now = time.time()


                t += 1.
                eta = 1 / (lamb * (t_0 + t))
                decay = decay * (1.0 - eta * lamb)

                #check if it is the end and need to store the model
                if inst_ind % 1000 == 0:
                    if args['op']:
                        err, gradient, Wcrf = par_objective(epoch, \
                            inst, rel_dict, Wv, b, We, Wcrf, args['d'] + args['len'], args['c'], len(vocab), \
                            rel_list, lambdas, trainer, inst_ind, eta, decay, True)
                    else:
                        err, gradient, Wcrf = par_objective(epoch, \
                            inst, rel_dict, Wv, b, We, Wcrf, args['d'], args['c'], len(vocab), \
                            rel_list, lambdas, trainer, inst_ind, eta, decay, True)
                else:
                    if args['op']:
                        err, gradient, Wcrf = par_objective(epoch, \
                            inst, rel_dict, Wv, b, We, Wcrf, args['d'] + args['len'], args['c'], len(vocab), \
                            rel_list, lambdas, trainer, inst_ind, eta, decay, False)
                    else:
                        err, gradient, Wcrf = par_objective(epoch, \
                            inst, rel_dict, Wv, b, We, Wcrf, args['d'], args['c'], len(vocab), \
                            rel_list, lambdas, trainer, inst_ind, eta, decay, False)
                        #gc.collect()
           
                grad_vec = roll_params_noWcrf(gradient, rel_list)
                update = ag.rescale_update(grad_vec)
                gradient = unroll_params_noWcrf(update, d, c, len(vocab), rel_list)
                
                for rel in rel_list:
                    rel_dict[rel] -= gradient[0][rel]
                Wv -= gradient[1]
                b -= gradient[2]
                We -= gradient[3]

                lstring = 'epoch: ' + str(epoch) + ' inst_ind: ' + str(inst_ind)\
                        + ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
                print lstring
                log.write(lstring + '\n')
                log.flush()

                epoch_error += err
                
                if inst_ind % 1000 == 0:
                    evaluate(epoch, inst_ind, test_trees, rel_dict, Wv, b, We, vocab, rel_list, d, c, mixed = False)

            Wcrf = Wcrf * decay
            # done with epoch
            print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
            lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                     + ' min error = ' + str(min_error) + '\n\n'
            log.write(lstring)
            log.flush()

            
            '''
            # save parameters if the current model is better than previous best model
            if epoch_error < min_error:
                min_error = epoch_error
                print 'saving model...'
                #params = unroll_params(r, args['d'], len(vocab), rel_list)
                
                if (args['op']):
                    params = unroll_params_crf(r, args['d'] + args['len'], args['c'], len(vocab), rel_list)
                else:
                    params = unroll_params_crf(r, args['d'], args['c'], len(vocab), rel_list)
                cPickle.dump( ( params, vocab, rel_list), paramfile)
                
                cPickle.dump( ( [rel_dict, Wv, b, We], vocab, rel_list), paramfile)
                
            else:
                os.remove(str(epoch)+'crf.model')
            '''
            

        #print 'saving model...'
        #cPickle.dump( ( [rel_dict, Wv, b, We], vocab, rel_list), paramfile)


    log.close()
    #paramfile.close()


