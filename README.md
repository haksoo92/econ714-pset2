# ECON-714: Problem Set I

## Problem 1: Github
The Github repo for the project is [here](https://github.com/haksoo92/econ714-pset-1).


## Problem 2: Integration
```python
integrateMidpoint(a, b, n, rrho, llambda)
integrateTrapezoid(a, b, n, rrho, llambda)
integrateSimpson(a, b, n, rrho, llambda)
```
define numerical integrations using the midpoint, trapezoid, Simpson's rules and a Monte Carlo.
The documentations are included in 
```bash
helloWorld.py
```

Figure 1 shows compute-time performance comparision with respect to the number of intervals in estimation. In particular, it shows that the trapezoid rule is the fastest and the Monte Carlo method the slowest.

Figure 2 shows numerical accuracy performance comparison with respect to the number of intervals in estimation. In particular, it shows that the Monte Carlo method is relatively inaccurate in lower N buy converges to the estimates of other methods as N grows.


## Problem 3
```python
minimizeNewtonRaphson_fd(objFct, grad_objFct, hess_objFct, x0, tol)
minimize_BFGS(objFct, grad_objFct, hess_objFct, x0, tol)
minimizeSD(objFct, grad_objFct, hess_objFct, x0, tol)
minimizeCD(objFct, grad_objFct, hess_objFct, x0, tol)
```
define numerical optimizers using the Newton-Raphson, BFGS, steepest descent, and conjugate descent methods.
The documentations and implementation details can be found in 
```bash
helloWorld.py
```
Figure 3 shows the computation time performance of each method. In particular, the figure shows the Newton-Raphson is the fastest and the steepest descent slowest when tolerance is sufficiently small. (I define the tolerance to be the maximal absolute distance between the updates. See the source code for implementation details.)

Figure 4 shows the numerical estimate for x. With respect to the accuracy of numerical estimates, the Newton-Raphson shows the fastest convergence and the BFGS method the slowest.




## Problem 4

```bash
pip install foobar
```

```python
import foobar
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
