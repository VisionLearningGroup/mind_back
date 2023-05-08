import pickle
import sys
import numpy as np

file = open(sys.argv[1], 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()
import pdb
pdb.set_trace()

import matplotlib.pyplot as plt
import numpy as np


for k, v in data.items():
    v = v.reshape(v.shape[0] * v.shape[1])
    plt.hist(v)
    plt.savefig(k+".png")
    plt.clf()


