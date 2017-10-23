from operator import itemgetter

# - an individual node contains the word associated with the node along with 
#   pointers to its kids and parents. 
class node:

    def __init__(self, word):
        if word != None:
            self.word = word
            self.kids = []
            self.parent = []
            self.finished = 0
            self.is_word = 1
            
            # add label
            self.trueLabel = 0

            # the "ind" variable stores the look-up index of the word in the 
            # word embedding matrix We. set this value when the vocabulary is finalized
            self.ind = -1

        else:
            self.is_word = 0

# - a dtree consists of a list of nodes
# - if you want to use a different dataset, check out the preprocessing scripts
#   that convert stanford dependency parses to dtrees
class dtree:

    def __init__(self, word_list):
        self.nodes = []
        for word in word_list:
            self.nodes.append(node(word))


    def add_edge(self, par, child, rel):
        self.nodes[par].kids.append( (child, rel ) )
        self.nodes[child].parent.append( (par, rel) )


    # return all non-None nodes
    def get_nodes(self):
        return [node for node in self.nodes if node.is_word]


    def get_node_inds(self):
        return [(ind, node) for ind, node in enumerate(self.nodes) if node.is_word]


    # get a node from the raw node list
    def get(self, ind):
        return self.nodes[ind]


    # return the raw text of the sentence
    def get_words(self):
        return ' '.join([node.word for node in self.get_nodes()[1:]])


    # return raw text of phrase associated with the given node
    def get_phrase(self, ind):

        node = self.get(ind)
        words = [(ind, node.word), ]
        to_do = []
        for ind, rel in node.kids:
            to_do.append(self.get(ind))
            words.append((ind, self.get(ind).word))

        while to_do:
            curr = to_do.pop()

            # add this kid's kids to to_do
            if len(curr.kids) > 0:
                for ind, rel in curr.kids:
                    words.append((ind, self.get(ind).word))
                    to_do.insert(0, self.get(ind))  


        return ' '.join([word for ind, word in sorted(words, key=itemgetter(0) ) ]).strip()  


    def reset_finished(self):
        for node in self.get_nodes():
            node.finished = 0


    # one tree's error is the sum of the error at all nodes of the tree
    def error(self):
        sum = 0.0
        for node in self.get_nodes():
            #sum += node.ans_error
            sum += node.label_error

        return sum
