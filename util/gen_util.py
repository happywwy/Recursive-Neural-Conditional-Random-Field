
import numpy as np

# - given a vector containing all parameters, return a list of unrolled parameters
# - specifically, these parameters are:
#   - rel_dict, dictionary of {dependency relation r: composition matrix W_r}
#   - Wv, the matrix for lifting a word embedding to the hidden space
#   - b, bias term
#   - We, the word embedding matrix
#   - Wc, classification matrix
#   - b_c, classification bias

def unroll_params(arr, d, c, len_voc, rel_list):

    mat_size = d * d
    #classification
    matClass_size = c * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size

    Wc = arr[ind : ind + matClass_size].reshape( (c, d) )
    ind += matClass_size

    b = arr[ind : ind + d].reshape( (d, 1) )
    ind += d

    b_c = arr[ind : ind + c].reshape( (c, 1) )
    ind += c

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, Wc, b, b_c, We]
    
#   similar to the above function, but exclude classification parameters

def unroll_params_noWcrf(arr, d, c, len_voc, rel_list):

    mat_size = d * d
    rel_dict = {}
    ind = 0

    for r in rel_list:
        rel_dict[r] = arr[ind: ind + mat_size].reshape( (d, d) )
        ind += mat_size

    Wv = arr[ind : ind + mat_size].reshape( (d, d) )
    ind += mat_size
   
    b = arr[ind : ind + d].reshape( (d, 1) )
    ind += d

    We = arr[ind : ind + len_voc * d].reshape( (d, len_voc))

    return [rel_dict, Wv, b, We]
    
# combine all parameters into a flat vector

def roll_params(params, rel_list):

    (rel_dict, Wv, Wc, b, b_c, We) = params

    rels = np.concatenate( [rel_dict[key].ravel() for key in rel_list] )

    return np.concatenate( (rels, Wv.ravel(), Wc.ravel(), b.ravel(), b_c.ravel(), We.ravel() ) )


def roll_params_noWcrf(params, rel_list):

    (rel_dict, Wv, b, We) = params

    rels = np.concatenate( [rel_dict[key].ravel() for key in rel_list] )

    return np.concatenate( (rels, Wv.ravel(), b.ravel(), We.ravel() ) )

# randomly initialize all parameters

def gen_dtrnn_params(d, c, rels):
    """
    Returns (dict{rels:[mat]}, Wv, Wc, b, b_c)
    """
    r = np.sqrt(6) / np.sqrt(2 * d + 1)
    r_Wc = 1.0 / np.sqrt(d)
    rel_dict = {}
    np.random.seed(3)
    for rel in rels:
	   rel_dict[rel] = np.random.rand(d, d) * 2 * r - r

    return (
	    rel_dict,
        #Wv
	    np.random.rand(d, d) * 2 * r - r,
        #Wc
        np.random.rand(c, d) * 2 * r_Wc - r_Wc,
        #b
	    np.zeros((d, 1)),
        #b_c
        np.random.rand(c, 1)
        )

 
#generate word embedding matrix
def gen_word_embeddings(d, total_num):

    for ind in range(total_num):
        word_vec = np.random.rand(d, 1)
        if ind == 0:
            word_embedding = word_vec
        else:
            word_embedding = np.c_[word_embedding, word_vec]
     
    return word_embedding


# returns list of zero gradients which backprop modifies, used for pretraining of RNN
def init_dtrnn_grads(rel_list, d, c, len_voc):

    rel_grads = {}
    for rel in rel_list:
	  rel_grads[rel] = np.zeros( (d, d) )

    return [
	    rel_grads,
	    np.zeros((d, d)),
        np.zeros((c, d)),
	    np.zeros((d, 1)),
        np.zeros((c, 1)),
	    np.zeros((d, len_voc))
	    ]
     
# returns list of zero gradients which backprop modifies, used for joint training of RNCRF
def init_crfrnn_grads(rel_list, d, c, len_voc):

    rel_grads = {}
    for rel in rel_list:
	  rel_grads[rel] = np.zeros( (d, d) )

    return [
	    rel_grads,
	    np.zeros((d, d)),
	    np.zeros((d, 1)),
	    np.zeros((d, len_voc))
	    ]
     

# random embedding matrix for gradient checks
def gen_rand_we(len_voc, d):
    r = np.sqrt(6) / np.sqrt(51)
    we = np.random.rand(d, len_voc) * 2 * r - r
    return we
