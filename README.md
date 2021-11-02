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


## Problem 3: Optimization
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
Figure 3 shows the computation time performance of each method. In particular, the figure shows the Newton-Raphson is the fastest and the steepest descent slowest when tolerance is sufficiently small. (I define the tolerance to be the maximal absolute distance between the updates of the optimizers. See the source code for implementation details.)

Figure 4 shows the numerical estimate for x. With respect to the accuracy of numerical estimates, the Newton-Raphson shows the fastest convergence and the BFGS method the slowest.


## Problem 4: Pareto Efficient Allocations




## Problem 5: Equilibrium Allocations
We obtain the equilibrium allocations using a nested optimization approach.

Each agent j solves
max_{x(1), ..., x(m)} sum_i alpha_i*(x(i)^(1+w(i))/(1+w(i)))
such that
sum_i p(i)*x(i) = sum_i p(i)*e(i).
The FOCs of the agent are
(alpha(k)*x(k)^w(k))/(alpha(1)*x(1)^w(1)) = p(k)/p(1), k=2,..., m
sum_i p(i)*x(i) = sum_i p(i)*e(i).

Given a price vector p, we can solve for the optimial allocations by minimizing the FOC errors. That is, by choosing x(i) that minimize sum_k ((alpha(k)*x(k)^w(k))/(alpha(1)*x(1)^w(1)) - p(k)/p(1))^2 + (sum_i p(i)*x(i) - sum_i p(i)*e(i))^2.

As such, we can obtain the equilibrium price vector as that clears the market. That is, we may obtain p such that
sum_j x(j,p) = sum_j e(j,i), i=1, ..., n.
by choosing p(i) that minimize 
sum_i (sum_j x(j,p) - sum_j e(j,i))^2.

The implementation details and test cases can be found in
```bash
helloWorld.py
```







## License
[MIT](https://choosealicense.com/licenses/mit/)
