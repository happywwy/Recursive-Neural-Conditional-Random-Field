# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:28:35 2015

@author: wangwenya
"""

"""
create tree structures from raw parses for training sentences
accumulate vocabulary
ignore lemmatization
differentiate beginning and inside of aspects/opinions
"""


from dtree_util import *
import gen_util as gen
import sys, cPickle, random, os
from numpy import *

# import dependency parse trees
f = open('data_semEval/raw_parses_sample', 'r')

indice = 0

data = f.readlines()
plist = []
tree_dict = []
vocab = []
rel_list = []

# import ground-truth aspect term labels and opinion term labels
label_file = open('data_semEval/aspectTerm_sample', 'r')
label_sentence = open('data_semEval/opinion_sample', 'r')

for line in data:
    if line.strip():
        rel_split = line.split('(')
        rel = rel_split[0]
        deps = rel_split[1][:-1]
        deps = deps.replace(')','')
        if len(rel_split) != 2:
            print 'error ', rel_split
            sys.exit(0)

        else:
            dep_split = deps.split(',')
            
        if len(dep_split) > 2:
            fixed = []
            half = ''
            for piece in dep_split:
                piece = piece.strip()
                if '-' not in piece:
                    half += piece

                else:
                    fixed.append(half + piece)
                    half = ''

                    #print 'fixed: ', fixed
            dep_split = fixed

        final_deps = []
        for dep in dep_split:
            words = dep.split('-')
            word = words[0]
            ind = int(words[len(words) - 1])

            if len(words) > 2:
                word = '-'.join([w for w in words[:-1]])

            final_deps.append( (ind, word.strip()) )
        # store dependency relations for each word pair    
        plist.append((rel,final_deps))

    # after processing one sentence
    else:
        max_ind = -1
        for rel, deps in plist:
            for ind, word in deps:
                if ind > max_ind:
                    max_ind = ind

        # load words into nodes, then make a dependency tree
        nodes = [None for i in range(0, max_ind + 1)]
        for rel, deps in plist:
            for ind, word in deps:
                nodes[ind] = word

        tree = dtree(nodes)

        opinion_words = []
            
        
        aspect_term = label_file.readline().rstrip()
        labeled_sent = label_sentence.readline().strip() #opinions
        
        aspect_BIO = {}
        
        #facilitate bio notation
        if '##' in labeled_sent:
                opinions = labeled_sent.split('##')[1].strip()
                opinions = opinions.split(',')
                
                for opinion in opinions:
                    op_list = opinion.split()[:-1]
                    if len(op_list) > 1:
                        for ind, term in enumerate(nodes):
                            if term != None:
                                if term == op_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] != None and nodes[ind + 1] == op_list[1]:
                                    tree.get(ind).trueLabel = 3
                                    for i in range(len(op_list) - 1):
                                        if nodes[ind + i + 1] != None and nodes[ind + i + 1] == op_list[i + 1]:
                                            tree.get(ind + i + 1).trueLabel = 4
                                        
                    elif len(op_list) == 1:
                        for ind, term in enumerate(nodes):
                            if term != None:
                                if term == op_list[0] and tree.get(ind).trueLabel == 0:
                                    tree.get(ind).trueLabel = 3
        
        if aspect_term != 'NIL':
            aspects = aspect_term.split(',')
            
                        
            #deal with same word but different labels
            for aspect in aspects:
                aspect = aspect.strip()
                #aspect is a phrase
                if ' ' in aspect:
                    aspect_list = aspect.split()
                    for ind, term in enumerate(nodes):
                        if term == aspect_list[0] and ind < len(nodes) - 1 and nodes[ind + 1] == aspect_list[1]:
                            tree.get(ind).trueLabel = 1
                            
                            for i in range(len(aspect_list) - 1):
                                if ind + i + 1 < len(nodes):
                                    if nodes[ind + i + 1] == aspect_list[i + 1]:
                                        tree.get(ind + i + 1).trueLabel = 2
                            break
                      
                #aspect is a single word
                else:
                    for ind, term in enumerate(nodes):
                        if term == aspect and tree.get(ind).trueLabel == 0:
                            tree.get(ind).trueLabel = 1
            
            
        # add dependency edges between nodes
        for rel, deps in plist:
            par_ind, par_word = deps[0]
            kid_ind, kid_word = deps[1]
            tree.add_edge(par_ind, kid_ind, rel)

        tree_dict.append(tree)  
        
        for node in tree.get_nodes():
            if node.word.lower() not in vocab:
                vocab.append(node.word.lower())
                
            node.ind = vocab.index(node.word.lower())
            
            for ind, rel in node.kids:
                if rel not in rel_list:
                    rel_list.append(rel)

        plist = []
        indice += 1



print 'rels: ', len(rel_list)
print 'vocab: ', len(vocab)

cPickle.dump((vocab, rel_list, tree_dict), open("data_semEval/final_input_sample", "wb"))



