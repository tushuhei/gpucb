# GP-UCB

A simple implementation of GP-UCB[1]. The main purpose is to provide the
overview of GP-UCB algorithm's dynamics.

## Usage
Please refer the bottom of `gpucb.py` for sample settings.
1. Set up a meshgrid which defines the search space.
```python
x = np.arange(-3, 3, 0.25)
y = np.arange(-3, 3, 0.25)
grid = np.meshgrid(x, y)
```

2. Set up an environment class which is equipped with `sample()` method.
```python
class DummyEnvironment(object):
    def sample(self, x):
        return np.sin(x[0]) + np.cos(x[1])
```

3. Create a GPUCB instance by padding the search grid and the environment
instance.
```python
env = DummyEnvironment()
agent = GPUCB(grid, env)
```

4. Iterate `learn()` method to obtain the estimated curve. If the search space
is 2D, you can also use `plot()` method to visualize the entire situation.
```python
for i in range(20):
    agent.learn()
    agent.plot()
```

![Sample Image](https://github.com/tushuhei/gpucb/blob/master/sample.gif)

## References
1. Srinivas, Niranjan, et al. "Gaussian process optimization in the bandit
setting: No regret and experimental design." arXiv preprint arXiv:0912.3995
(2009).
