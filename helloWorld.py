import math
from re import S
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import outer
from numpy.linalg import eig
import scipy
from scipy.sparse import dok

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
import numdifftools as nd

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
    # print(x1)
    
    if abs(x1[0,0] - x0[0,0]) < tol and abs(x1[1,0] - x0[1,0]) < tol:
        return x0
    else:
        return minimizeNewtonRaphson(f, df, ddf, x1, tol)

x0 = np.array([[-100], [-100]])
tol = 1e-16
print(minimizeNewtonRaphson(eq, grad_obj, hess_obj, x0, tol))


# Define Newton-Raphson optimizer (finite difference)
def minimizeNewtonRaphson_fd(f, df, ddf, x0, tol):
    df_x0 = df(x0)
    ddf_x0 = ddf(x0)
    inv_ddf_x0 = np.linalg.inv(ddf_x0)

    x1 = x0 - np.transpose(np.matmul(inv_ddf_x0, np.transpose(df_x0)))
    # print(x1)
    
    if abs(x1[0] - x0[0]) < tol and abs(x1[1] - x0[1]) < tol:
        return x0
    else:
        return minimizeNewtonRaphson_fd(f, df, ddf, x1, tol)

objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
hess_objFct = nd.Hessian(objFct)
x0 = [-1e3, -1e3]
tol = 1e-16
print(minimizeNewtonRaphson_fd(objFct, grad_objFct, hess_objFct, x0, tol))


# .........................................................
# (2) BFGS
def minimize_BFGS(objFct, grad_objFct, hess_objFct, x0, tol):
    g0 = grad_objFct(x0)
    Q0 = hess_objFct(x0)
    H0 = np.linalg.inv(Q0)
    while True:
        x1 = x0.reshape(2,1) - (H0 @ g0.reshape(2,1))
        p0 = x1.reshape(2,1) - x0.reshape(2,1)
        g1 = grad_objFct(x1)
        q0 = g1.reshape(2,1) - g0.reshape(2,1)
        H1 = H0 + (1 + (q0.T@H0@q0)/(p0.T@q0))*((p0@p0.T)/(p0.T@q0)) - ((p0@q0.T@H0)+ (H0@q0@p0.T))/(q0.T@p0)
        if abs(x1[0] - x0[0]) < tol and abs(x1[1] - x0[1]) < tol:
            break
        else:
            x0, H0, g0 = x1, H1, g1
    return x1

objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
hess_objFct = nd.Hessian(objFct)
x0 = np.array([-2,2])
tol = 1e-12
print(minimize_BFGS(objFct, grad_objFct, hess_objFct, x0, tol))

# .........................................................
# (3) Steepest descent
from scipy.optimize import minimize

def minimizeSD(objFct, grad_objFct, x0, tol):
    while True:
        d0 = -grad_objFct(x0)

        innerObjFct = lambda a : objFct(x0 + (a*d0))

        a0 = np.array([0])
        aalpha0 = minimize(innerObjFct, a0, method='BFGS')
        aalpha0 = aalpha0.x

        x1 = x0 + (aalpha0*d0)
        if abs(x1[0] - x0[0]) < tol and abs(x1[1] - x0[1]) < tol:
            break
        else:
            x0 = x1
    return x1

objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
x0 = np.array([-2,2])
tol = 1e-6
print(minimizeSD(objFct, grad_objFct, x0, tol))


# .........................................................
# (4) Conjugate descent
objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
# hess_objFct = nd.Hessian(objFct)

def minimizeCD(objFct, grad_objFct, x0, tol):
    r0 = -grad_objFct(x0)
    d0 = r0
    while True:
        innerObjFct = lambda a : objFct(x0 + (a*d0))

        a0 = np.array([0])
        aalpha0 = minimize(innerObjFct, a0, method='BFGS')
        aalpha0 = aalpha0.x

        x1 = x0 + (aalpha0*d0)
        r1 = -grad_objFct(x1)
        beta1 = (r1.reshape(2,1).T @ r1.reshape(2,1))/(r0.reshape(2,1).T @ r0.reshape(2,1))
        beta1 = beta1[0,0]
        d1 = r1 + beta1*d0

        if abs(x1[0] - x0[0]) < tol and abs(x1[1] - x0[1]) < tol:
            break
        else:
            x0, d0, r0 = x1, d1, r1
    return x1

x0 = np.array([-2,2])
tol = 1e-6
print(minimizeCD(objFct, grad_objFct, x0, tol))


# =========================================================
# Problem 4: Pareto Efficient Allocations
# =========================================================
def util_i(x, aalpha, oomega, m, n):
    sum_util = 0
    for i in range(m):
        curr_util = aalpha[i]*((x[i]**(1+oomega[i]))/(1+oomega[i]))
        sum_util = sum_util + curr_util
    return sum_util

# Test
m, n = 3, 3 # m = number of goods; n = number of agents
x_i = np.array([0.2, 0.2, 0.6])
aalpha = np.array([0.8, 0.1, 0.1]) # common acrross agents
oomega_i = np.array([-0.3, -0.4, -0.3])

util_i(x_i, aalpha, oomega_i, m, n)

def  util_social(X, Aalpha, Oomega, m, n, llambda):
    sum_util_social = 0
    for j in range(n):
        curr_x_i = X[j,:]
        curr_aalpha_i = Aalpha[j,:]
        curr_oomega_i = Oomega[j,:]
        curr_util_i = llambda[j]*util_i(curr_x_i, curr_aalpha_i, curr_oomega_i, m, n)
        sum_util_social = sum_util_social + curr_util_i
        # print(curr_util_i)
    return sum_util_social

# Test
X = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.8, 0.2, 0.6]])
Aalpha = np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
Oomega = np.array([[-0.3, -0.4, -0.3], [-0.3, -0.4, -0.3], [-0.3, -0.4, -0.3]])
llambda = np.array([0.1, 0.3, 0.6])

util_social(X, Aalpha, Oomega, m, n, llambda)

def lagrangian_obj(X_vec, lm_vec, llambda_vec, Aalpha, Oomega, Endow, m, n):
    X = X_vec.reshape(n,m)
    llambda = llambda_vec
    lm = lm_vec
    outer_sum = 0
    for i in range(m): # product
        inner_sum = 0
        for j in range(n): # agent
            inner_sum = inner_sum + (Endow[i,j] - X[i,j])
        outer_sum = outer_sum + lm[i]*inner_sum
        # print("outersum")
        # print(outer_sum)
    return util_social(X, Aalpha, Oomega, m, n, llambda) + outer_sum

# Test
X_vec = X.reshape(m*n,)
lm_vec = np.array([0.1, 0.1, 0.1])
llambda_vec = np.array([0.1, 0.3, 0.6])
Endow = np.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.8, 0.4, 0.2]])

lagrangian_obj(X_vec, lm_vec, llambda_vec, Aalpha, Oomega, Endow, m, n)

wrapped_lagrangian = lambda x : lagrangian_obj(x[0:m*n], x[m*n:], llambda_vec, Aalpha, Oomega, Endow, m, n)
grad_wrapped_lagrangian = nd.Gradient(wrapped_lagrangian)

objfct_pareto = lambda x: np.sum(grad_wrapped_lagrangian(x)**2)



# Test
x0 = np.ones((12,))*(5)
tol = 1e-2
res = minimize(objfct_pareto, x0, method='BFGS')



# =========================================================
# Problem 5: Equilibrium Allocation
# =========================================================
import numpy.matlib

def getFocErr(price_vec, x_vec, e_vec, aalpha, oomega):
    foc_err = 0
    for k in range(m-1):
        curr_foc_err = (((aalpha[k+1]*(x_vec[k+1]**oomega[k+1]))/price_vec[k+1]) - ((aalpha[0]*(x_vec[0]**oomega[0]))/price_vec[0]))**2
        foc_err = foc_err + curr_foc_err
    foc_err = foc_err + (np.dot(price_vec, x_vec) - np.dot(price_vec, e_vec))**2
    return foc_err

def getExcessDemand(price_vec, e_vec, aalpha, oomega, m, n):
    """
    price_vec = (m,)
    aalpha = (m,)
    oomega = (m,)
    """
    objfct = lambda x_vec: getFocErr(price_vec, x_vec, e_vec, aalpha, oomega)

    x_vec0 = np.ones((m,))
    res = minimize(objfct, x_vec0, method='BFGS')
    x_vec1 = res.x

    return x_vec1

# Test
price_vec = np.array([2, 2, 2])
e_vec = np.array([0.5, 1, 1])
aalpha = np.array([1/3, 1/3, 1/3])
oomega = -np.array([0.9, 0.1, 0.0])

getExcessDemand(price_vec, e_vec, aalpha, oomega, m, n)

def getPriceFocErr(price_vec, E, Aalpha, Oomega, m, n):
    sum_focerr = 0
    for i in range(m):
        sum_exdemand = 0
        for j in range(n):
            curr_aalpha = Aalpha[j,:]
            curr_oomega = Oomega[j,:]
            curr_exdemand = getExcessDemand(price_vec, e_vec, curr_aalpha, curr_oomega, m, n)
            sum_exdemand = sum_exdemand + curr_exdemand
        curr_focerr = (sum_exdemand[i] - np.sum(E[:,i]))**2
        sum_focerr = sum_focerr + curr_focerr
    return sum_focerr

E = np.matlib.repmat(e_vec, 3, 1)
Aalpha = np.matlib.repmat(aalpha, 3, 1)
Oomega = np.matlib.repmat(oomega, 3, 1)
price_vec = np.array([2, 2, 2])
getPriceFocErr(price_vec, E, Aalpha, Oomega, m, n)

def getEqPrice(E, Aalpha, Oomega, m, n):
    objfct = lambda price_vec: getPriceFocErr(price_vec, E, Aalpha, Oomega, m, n)
    p_vec0 = np.ones((m,))
    res = minimize(objfct, p_vec0, method='BFGS')
    p_vec1 = res.x
    return p_vec1

# Test
getEqPrice(E, Aalpha, Oomega, m, n)






    
"""
def lagrangian_demand(x_vec, price_vec, aalpha, oomega, endow, m, n):
    X = X_vec.reshape(n,m)
    llambda = llambda_vec
    outer_sum = 0
    for i in range(m): # product
        inner_sum = 0
        for j in range(n): # agent
            inner_sum = inner_sum + (Endow[i,j] - X[i,j])
        outer_sum = outer_sum + llambda[i]*inner_sum
        # print("outersum")
        # print(outer_sum)
    return util_social(X, Aalpha, Oomega, m, n, llambda) + outer_sum
"""
