import os
import numpy as np
dir='./save/mnist_mlp/24/iid/'
if os.path.exists('./save/mnist_mlp/24/iid/50e_w_locals.npy'):
    w_locals = np.load(dir+'/50e_w_locals.npy',allow_pickle=True)[()]
    print(w_locals)
    