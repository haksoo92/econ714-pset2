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
minimizeBFGS(objFct, grad_objFct, hess_objFct, x0, tol)
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
The social planner solves
```math
max_{x(j,i)} sum_{j} lambda(j) u(j, x(j,1), ..., x(j,m))
sum_{j} x(j,i) = sum_{j} e(j,i), i=1,...,m
```
so that i indexes the goods j indexes households.
Substituting in the constraints, we may instead solve an unconstrained objective. 
In particular, we may substitute in
```math
x(m,i) = sum_j e(j,i) - sum_{k \neq m} x(k,i).
```
I use the BFGS algorithm to find the maximum of the objective.

### Symmetric Case
First, I consider the following symmetric case:
```python
Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
Oomega = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]])
llambda = np.array([0.5, 0.3, 0.2])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
```
And we obtain that the Pareto efficient allocations are
```python
array([[0.98684766, 0.98684766, 0.98684763],
       [0.35525783, 0.35525783, 0.35525786],
       [0.15789451, 0.15789451, 0.15789451]])
```

### Asymmetric Case
Next, I consider the following asymmetric case in which agents are heterogeneous:
```python
Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
Oomega = np.array([[-0.6, -0.5, -0.4], [-0.5, -0.5, -0.5], [-0.4, -0.5, -0.6]])
llambda = np.array([0.5, 0.3, 0.2])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
```
And we obtain that the Pareto efficient allocations are
```python
array([[1.02446797, 0.98683915, 0.9462659 ],
       [0.37060194, 0.35526582, 0.34443121],
       [0.10493009, 0.15789503, 0.20930289]])
```

The BFGS algorithm finds the optima quickly (less than 2 seconds). Yet, depending on initial values, it may fail to converge. To avoid convergence failures, one may use converged allocations in a model with similar parameters as initial values. 

As expected, heterogeneity in preferences across goods (omega) affects the Pareto efficient allocations. In particular, the smaller is the omega of a good, the greater is the allocation to that good for a given individual, ceteris paribus.

Unfortunately, the method does not succeed when m=n=10.

## Problem 5: Equilibrium Allocations
We obtain the equilibrium allocations using a nested optimization approach.

Each agent j solves
```math
max_{x(1), ..., x(m)} sum_i alpha_i*(x(i)^(1+w(i))/(1+w(i))) such that
sum_i p(i)*x(i) = sum_i p(i)*e(i).
```

The FOCs of the agent are
```math
(alpha(k)*x(k)^w(k))/(alpha(1)*x(1)^w(1)) = p(k)/p(1), k=2,..., m
sum_i p(i)*x(i) = sum_i p(i)*e(i).
```

Given a price vector p, we can solve for the optimial allocations by minimizing the FOC errors. That is, by choosing x(i) that minimize 
```math
sum_k ((alpha(k)*x(k)^w(k))/(alpha(1)*x(1)^w(1)) - p(k)/p(1))^2 + (sum_i p(i)*x(i) - sum_i p(i)*e(i))^2.
```

As such, we can obtain the equilibrium price vector as that clears the market. That is, we may obtain p such that
```math
sum_j x(j,p) = sum_j e(j,i), i=1, ..., n.
```
by choosing p(i) that minimize 
```math
sum_i (sum_j x(j,p) - sum_j e(j,i))^2.
```

The implementation details and test cases can be found in
```bash
helloWorld.py
```







## License
[MIT](https://choosealicense.com/licenses/mit/)
