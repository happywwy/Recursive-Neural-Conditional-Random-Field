learn -a "l2sgd" -p max_iterations=1  -m CoNLL2000.model crfnn_train_full.txt

tag -t -m CoNLL2000.model crfnn_test.txt