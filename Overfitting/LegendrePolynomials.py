from scipy.special import legendre
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

n = 3
Pn = legendre(n)
x = np.arange(-1,1,0.1)
y = Pn(x)

min = -1.0
max = 1.0
step = 0.05

for n in range(6):
    Pn = legendre(n)
    x = np.arange(min,max+step,step)
    y = Pn(x)
    plt.plot(x, y)

plt.xlim(-1.0,1.0)
plt.ylim(-1.0,1.01)

plt.savefig('legendre_polynomes.png')
plt.show()