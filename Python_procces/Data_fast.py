import numpy as np
d=np.load('pics_todos_ventanas.npz')
print((d['y']==1).sum(), (d['y']==0).sum())