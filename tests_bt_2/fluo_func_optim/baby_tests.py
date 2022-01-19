import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
np.random.seed(237)


noise_level = 0.1

def f(x, noise_level=noise_level):
    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level


res = gp_minimize(
    f,                   # the function to minimize
    [(-2.0, 2.0)],       # the bounds on each dimension of x
    acq_func="EI",       # the acquisition function
    n_calls=15,          # the number of evaluations of f
    n_random_starts=5,   # the number of random intialization points
    noise=0.1**2,        # the noise level (optional)
    random_state=1234    # the random seed
)

print(f"x* = {res.x[0]}, f(x*) = {res.fun}")

x = np.linspace(-2, 2, 400).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in x]
plt.plot(x, fx, "r--", label="True (unkown)")
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                         [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
         alpha=.2, fc="r", ec="None")
plt.legend()
plt.grid()
plt.show()

# oker :)
