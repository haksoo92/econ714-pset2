import math
import numpy as np

# =========================================================
# Problem 2: Integration
# =========================================================
# Compute \int_0^T exp(-\rho t) u(1-exp(-\lambda t)) dt
# T = 100, \rho = 0.04, \lambda = 0.02, u(c) = -exp(-c)

# Set parameters
T = 100
rrho = 0.04
llambda = 0.02


# Compute utility given consumption
def utilityFct(cc):
    return -math.exp(-cc)


# .........................................................
# Midpoint
a = 0
b = T
n = 100 # number of intervals

# Define the function to integrate
def integrandFct(t, rrho, llambda):
    return math.exp(-rrho*t)*utilityFct(1-math.exp(-llambda*t))


# Compute step size
def getStepSize(a, b, n):
    return (b-a)/n

# Numerically integrate using the midpoint rule
def integrateMidpoint(a, b, n, rrho, llambda):
    currSum = 0
    h = getStepSize(a, b, n)
    for j in range(1,n+1):
        currEvalPoint = a + ((j-0.5)*h)
        currArea = integrandFct(currEvalPoint, rrho, llambda)
        currSum = currSum + currArea
    return currSum

print("Midpoint Rule Estimate: "+ str(integrateMidpoint(a, b, n, rrho, llambda)))

# .........................................................
# Trapezoid

# Numerically integrate using the trapezoid rule
# !Typo in the notes NM_1 Slide 22: f is evaluated at a + h*j
def integrateTrapezoid(a, b, n, rrho, llambda):
    h = getStepSize(a, b, n)
    currSum = 0
    for j in range(1,n): # loop to n-1
        currEvalPoint = a + (j*h)
        currArea = integrandFct(currEvalPoint, rrho, llambda)
        currSum = currSum + currArea
    return h*(currSum + 0.5*(integrandFct(a, rrho, llambda) + integrandFct(b, rrho, llambda)))

print("Trapezoid Rule Estimate: "+ str(integrateTrapezoid(a, b, n, rrho, llambda)))


# .........................................................
# Simpson

# Get evaluation points for the trapezoid and Simpson's Rules
def getEvalPoint(a, h, j):
    return a + (j*h)

# Numerically integrate using the trapezoid rule
# Assumes n is even
def integrateSimpson(a, b, n, rrho, llambda):
    h = getStepSize(a, b, n)
    currSum1 = 0
    currSum2 = 0
    for j in range(1, int(n/2)):
        currEvalPoint1 = getEvalPoint(a, h, 2*j)
        currSum1 = currSum1 + integrandFct(currEvalPoint1, rrho, llambda)
    for j in range(1, int(n/2+1)):
        currEvalPoint2 = getEvalPoint(a, h, (2*j)-1)
        currSum2 = currSum2 + integrandFct(currEvalPoint2, rrho, llambda)
    return (h/3)*(integrandFct(a, rrho, llambda) + (2*currSum1) + (4*currSum2) + integrandFct(b, rrho, llambda)) 

print("Simpson Rule Estimate: "+ str(integrateSimpson(a, b, n, rrho, llambda)))

# .........................................................
# Monte Carlo

# Numerically integrate using the trapezoid rule
# Assumes n is even
def integrateMonteCarlo(a, b, n, rrho, llambda):
    # Get uniform draws from [0,T]
    x_vec = np.random.uniform(low=0, high=b, size=n)
    currSum = 0
    for j in range(n):
        currEvalPoint = x_vec[j]
        currSum = currSum + integrandFct(currEvalPoint, rrho, llambda)
    return (b/n)*currSum

print("Monte Carlo Estimate: "+ str(integrateMonteCarlo(a, b, n, rrho, llambda)))


# .........................................................
# Performance Comparison

# TODO: Write the compute time function that takes as input function to compute
#       to compute the time-to-compute


# =========================================================
# Problem 3: Optimization
# =========================================================
# Optimize min_{x,y} 100*(y-x^2)^2 + (1-x)^2
# using (1) Newton-Raphson, (2) BFGS, (3) steepest descent, (4) conjugate descent
from sympy.abc import x, y
from sympy import ordered, Matrix, hessian

# Define objective function
eq = 100*((y-(x**2))**2) + (1-x)**2
v = list(ordered(eq.free_symbols)); v

# Define gradient function
gradient = lambda f, v: Matrix([f]).jacobian(v)

# Symbolic gradient and hessian
grad_obj = gradient(eq, v)
hess_obj = hessian(eq, v)

def getSubstitued(symf, evalx, evaly):
    subx = symf.subs(x, evalx)
    subxy = subx.subs(y, evaly)
    return subxy

# .........................................................
# (1) Newton-Raphson

# Define Newton-Raphson optimizer (symbolic)
def minimizeNewtonRaphson(f, df, ddf, x0, tol):
    df_x0 = getSubstitued(df, x0[0,0], x0[1,0])
    inv_ddf_x0 = getSubstitued(ddf.inv(), x0[0,0], x0[1,0])
    # inv_ddf_x0 = np.linalg.inv(ddf_x0)

    df_x0 = np.array(df_x0).astype(np.float128)
    inv_ddf_x0 = np.array(inv_ddf_x0).astype(np.float128)

    x1 = x0 - np.matmul(inv_ddf_x0, np.transpose(df_x0))
    print(x1)
    
    if abs(x1[0,0] - x0[0,0]) + abs(x1[1,0] - x0[1,0]) < tol:
        return x0
    else:
        return minimizeNewtonRaphson(f, df, ddf, x1, tol)

x0 = np.array([[-100], [-100]])
tol = 1e-16
print(minimizeNewtonRaphson(eq, grad_obj, hess_obj, x0, tol))


# .........................................................
# (2) BFGS


"""
# Define objective function
def objFct(x,y):
    return 100*((y-(x**2))**2) + (1-x)**2


"""
