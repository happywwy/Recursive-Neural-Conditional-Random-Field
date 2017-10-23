# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:37:53 2015

@author: wangwenya
"""

"""
parse the preprocessed sentences to dependency formats
"""


import subprocess

#   stanford dependency parser to create a dependency parse tree for each sentence
out_file = open('./data_semEval/raw_parses_sample', 'w')

# change these paths to point to your stanford parser and the path for your preprocessed sentences
p = subprocess.Popen(["bash","lexparser.sh","./data_semEval/sample.txt"], stdout=subprocess.PIPE)
output, err = p.communicate()


for line in output:
    out_file.write(line)
    
out_file.close()
