import numpy as np
gradients = [[np.zeros(15), 0.5], [np.ones(15), -1.0]]

grad, rew = zip(*gradients)
# grad = gradients[:,0]
print(grad, rew)