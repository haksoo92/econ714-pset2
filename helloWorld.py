import math
from re import S
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import outer
from numpy.linalg import eig
import scipy
from scipy.sparse import dok
import time
import matplotlib.pyplot as plt

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


def integrateMidpoint(a, b, n, rrho, llambda):
    """
    Numerically integrate using the midpoint rule

    :param a: lower bound of the integral
    :param b: upper bound of the integral
    :param n: number of intervals
    :param rrho: problem parameter
    :param llambda: problem parameter
    :return: numerical integral
    """
    currSum = 0
    h = getStepSize(a, b, n)
    for j in range(1,n+1):
        currEvalPoint = a + ((j-0.5)*h)
        currArea = h*integrandFct(currEvalPoint, rrho, llambda)
        currSum = currSum + currArea
    return currSum

print("Midpoint Rule Estimate: "+ str(integrateMidpoint(a, b, n, rrho, llambda)))

# .........................................................
# Trapezoid



# !Typo in the notes NM_1 Slide 22: f is evaluated at a + h*j
def integrateTrapezoid(a, b, n, rrho, llambda):
    """
    Numerically integrate using the trapezoid rule

    :param a: lower bound of the integral
    :param b: upper bound of the integral
    :param n: number of intervals
    :param rrho: problem parameter
    :param llambda: problem parameter
    :return: numerical integral
    """
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


def integrateSimpson(a, b, n, rrho, llambda):
    """
    Numerically integrate using the Simpson's rule

    :param a: lower bound of the integral
    :param b: upper bound of the integral
    :param n: number of intervals
    :param rrho: problem parameter
    :param llambda: problem parameter
    :return: numerical integral
    """
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
    """
    Numerically integrate using the Monte Carlo method

    :param a: lower bound of the integral
    :param b: upper bound of the integral
    :param n: number of intervals
    :param rrho: problem parameter
    :param llambda: problem parameter
    :return: numerical integral
    """
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

# Write the compute time function that takes as input function to compute
#       to compute the time-to-compute
dat_n= []
dat_t_mid = []
dat_t_trap = []
dat_t_simpson = []
dat_t_monte = []
dat_y_mid = []
dat_y_trap = []
dat_y_simpson = []
dat_y_monte = []


def getPerformance(f, a, b, n, rrho, llambda):
    start = time.time()
    estval = f(a, b, n, rrho, llambda)
    end = time.time()
    return (end-start, estval)

# Test
t, val = getPerformance(integrateMonteCarlo, a, b, n, rrho, llambda)

NN = 1e6
curr_n = 1e2
dt = 1e1
loop_st = time.time()
while True:
    n = int(curr_n)
    t_mid, val_mid = getPerformance(integrateMidpoint, a, b, n, rrho, llambda)
    t_trap, val_trap = getPerformance(integrateTrapezoid, a, b, n, rrho, llambda)
    t_simp, val_simp = getPerformance(integrateSimpson, a, b, n, rrho, llambda)
    t_monte, val_monte = getPerformance(integrateMonteCarlo, a, b, n, rrho, llambda)

    dat_n.append(curr_n)

    dat_t_mid.append(t_mid)
    dat_t_trap.append(t_trap)
    dat_t_simpson.append(t_simp)
    dat_t_monte.append(t_monte)    

    dat_y_mid.append(val_mid)
    dat_y_trap.append(val_trap)
    dat_y_simpson.append(val_simp)
    dat_y_monte.append(val_monte)

    curr_n = curr_n*dt

    loop_et = time.time()

    print(curr_n)
    if curr_n > NN or loop_et - loop_st > 20:
        break

plt.plot(dat_n, dat_t_mid)
plt.plot(dat_n, dat_t_trap)
plt.plot(dat_n, dat_t_simpson)
plt.plot(dat_n, dat_t_monte)
plt.xscale('log')
plt.title('Peformance Comparison')
plt.ylabel('Compute Time (seconds)')
plt.xlabel('N')
plt.legend(['Midpoint', 'Trapezoid', 'Simpson', 'Monte Carlo'])
plt.show()

plt.plot(dat_n, dat_y_mid)
plt.plot(dat_n, dat_y_trap)
plt.plot(dat_n, dat_y_simpson)
plt.plot(dat_n, dat_y_monte)
plt.xscale('log')
plt.title('Peformance Comparison')
plt.ylabel('Numerical Estimates (seconds)')
plt.xlabel('N')
plt.legend(['Midpoint', 'Trapezoid', 'Simpson', 'Monte Carlo'])
plt.show()


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
    """
    Minimize the objective function f using the Newton-Raphson method

    :param f: the objective function
    :param df: the gradient of the objective (nd object)
    :param ddf: the hessian of the objective (nd object)
    :param x0: the initial point
    :param tol: tolerance
    :return: numerical minimizer
    """
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
    """
    Minimize the objective function f using the BFGS method

    :param f: the objective function
    :param df: the gradient of the objective (nd object)
    :param ddf: the hessian of the objective (nd object)
    :param x0: the initial point
    :param tol: tolerance
    :return: numerical minimizer
    """
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

def minimizeSD(objFct, grad_objFct, hess_objFct, x0, tol):
    """
    Minimize the objective function f using the steepest descent method

    :param f: the objective function
    :param df: the gradient of the objective (nd object)
    :param ddf: the hessian of the objective (nd object)
    :param x0: the initial point
    :param tol: tolerance
    :return: numerical minimizer
    """
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
print(minimizeSD(objFct, grad_objFct, hess_objFct, x0, tol))


# .........................................................
# (4) Conjugate descent
objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
hess_objFct = nd.Hessian(objFct)

def minimizeCD(objFct, grad_objFct, hess_objFct, x0, tol):
    """
    Minimize the objective function f using the conjugate descent method

    :param f: the objective function
    :param df: the gradient of the objective (nd object)
    :param ddf: the hessian of the objective (nd object)
    :param x0: the initial point
    :param tol: tolerance
    :return: numerical minimizer
    """
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
print(minimizeCD(objFct, grad_objFct, hess_objFct, x0, tol))

# Performance comparison
dat_tol= []
dat_t_newton = []
dat_t_bfgs = []
dat_t_sd = []
dat_t_cd = []
dat_y1_newton = []
dat_y2_newton = []
dat_y1_bfgs = []
dat_y2_bfgs = []
dat_y1_sd = []
dat_y2_sd = []
dat_y1_cd = []
dat_y2_cd = []

def getPerformance3(f, objFct, grad_objFct, hess_objFct, x0, tol):
    start = time.time()
    estval = f(objFct, grad_objFct, hess_objFct, x0, tol)
    end = time.time()
    return (end-start, estval)

# Test
objFct = lambda x : 100*((x[1]-(x[0]**2))**2) + (1.-x[0])**2
grad_objFct = nd.Gradient(objFct)
hess_objFct = nd.Hessian(objFct)
x0 = np.array([-2,2])
tol = 1e-12
t, val = getPerformance3(minimize_BFGS, objFct, grad_objFct, hess_objFct, x0, tol)

max_tol = 1e-10
curr_tol = 1e-1
dt = 1e-1
loop_st = time.time()
while True:
    tol = curr_tol
    t_newton, val_newton = getPerformance3(minimizeNewtonRaphson_fd, objFct, grad_objFct, hess_objFct, x0, tol)
    t_bfgs, val_bfgs = getPerformance3(minimize_BFGS, objFct, grad_objFct, hess_objFct, x0, tol)
    t_sd, val_sd = getPerformance3(minimizeSD, objFct, grad_objFct, hess_objFct, x0, tol)
    t_cd, val_cd = getPerformance3(minimizeCD, objFct, grad_objFct, hess_objFct, x0, tol)

    dat_tol.append(curr_tol)

    dat_t_newton.append(t_newton)
    dat_t_bfgs.append(t_bfgs)
    dat_t_sd.append(t_sd)
    dat_t_cd.append(t_cd)    

    dat_y1_newton.append(val_newton[0])
    dat_y2_newton.append(val_newton[1])
    dat_y1_bfgs.append(val_bfgs[0,0])
    dat_y2_bfgs.append(val_bfgs[1,0])
    dat_y1_sd.append(val_sd[0])
    dat_y2_sd.append(val_sd[1])
    dat_y1_cd.append(val_cd[0])
    dat_y2_cd.append(val_cd[1])
    
    curr_tol = curr_tol*dt

    loop_et = time.time()

    if curr_tol <= max_tol:
        break

plt.plot(dat_tol, dat_t_newton)
plt.plot(dat_tol, dat_t_bfgs)
plt.plot(dat_tol, dat_t_sd)
plt.plot(dat_tol, dat_t_cd)
plt.xscale('log')
plt.title('Peformance Comparison')
plt.ylabel('Compute Time (seconds)')
plt.xlabel('Tolerance')
plt.legend(['Newton-Raphson', 'BFGS', 'Steepest Descent', 'Conjugate Descent'])
plt.show()

plt.plot(dat_tol, dat_y1_newton)
plt.plot(dat_tol, dat_y1_bfgs)
plt.plot(dat_tol, dat_y1_sd)
plt.plot(dat_tol, dat_y1_cd)
plt.xscale('log')
plt.title('Peformance Comparison')
plt.ylabel('Numerical Estimates')
plt.xlabel('Tolerance')
plt.legend(['Newton-Raphson', 'BFGS', 'Steepest Descent', 'Conjugate Descent'])
plt.show()


# =========================================================
# Problem 4: Pareto Efficient Allocations
# =========================================================
def util_i(x, aalpha, oomega, m, n):
    """
    Computes the agent i's utility given model parameters

    :param x: a vector of allocations
    :param aalpha: a vector of goods weights
    :param oomega: a vector of elasticities across goods
    :param m: number of goods
    :param n: nubmer of individuals
    :return: utility of an agent
    """
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
    """
    Computes the social utility

    :param X: a matrix of allocations (j=hh,i=good)
    :param Aalpha: a matrix of goods weights
    :param Oomega: a matrix of elasticities across goods
    :param m: number of goods
    :param n: nubmer of individuals
    :param llambda: a vector of social weights
    :return: social utility
    """
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

# ................
# Symmetric Case I
# ................

# X = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.8, 0.2, 0.6]])
Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
# Oomega = np.array([[-0.3, -0.9, -0.3], [-0.3, -0.9, -0.3], [-0.3, -0.9, -0.3]])
Oomega = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]])
llambda = np.array([0.5, 0.3, 0.2])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

def pareto_obj(X_vec, llambda_vec, Aalpha, Oomega, Endow, m, n):
    """
    Substitutes in the constraints so that X_vec is of dimension (n-1) by m
    """
    X = X_vec.reshape(n-1,m)
    X_sum = np.sum(X, axis=0)
    endow_sum = np.sum(Endow, axis=0)
    X = np.append(X, [endow_sum-X_sum], axis=0)
    llambda = llambda_vec
    return util_social(X, Aalpha, Oomega, m, n, llambda)

# Test
X = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6]])
X_vec = X.reshape(m*(n-1),)

llambda_vec = llambda
-pareto_obj(X_vec, llambda_vec, Aalpha, Oomega, Endow, m, n)

obj = lambda x : -pareto_obj(x, llambda_vec, Aalpha, Oomega, Endow, m, n)

x0 = np.ones((m*(n-1),))*.5
res = minimize(obj, x0, method='BFGS')

Xres_vec = res.x
Xres = Xres_vec.reshape(n-1,m)
Xres_sum = np.sum(Xres, axis=0)
endow_sum = np.sum(Endow, axis=0)
Xres = np.append(Xres, [endow_sum- Xres_sum], axis=0) # pareto optimal allocation
Xres

# ..................
# Asymmetric Case II
# ..................

# X = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6], [0.8, 0.2, 0.6]])
Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
# Oomega = np.array([[-0.6, -0.5, -0.4], [-0.5, -0.5, -0.5], [-0.4, -0.5, -0.6]])
Oomega = np.array([[-0.8, -0.5, -0.2], [-0.5, -0.5, -0.5], [-0.4, -0.5, -0.6]])
llambda = np.array([0.5, 0.3, 0.2])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

def pareto_obj(X_vec, llambda_vec, Aalpha, Oomega, Endow, m, n):
    X = X_vec.reshape(n-1,m)
    X_sum = np.sum(X, axis=0)
    endow_sum = np.sum(Endow, axis=0)
    X = np.append(X, [endow_sum-X_sum], axis=0)
    llambda = llambda_vec
    return util_social(X, Aalpha, Oomega, m, n, llambda)

# Test
X = np.array([[0.2, 0.2, 0.6], [0.2, 0.2, 0.6]])
X_vec = X.reshape(m*(n-1),)

llambda_vec = llambda
-pareto_obj(X_vec, llambda_vec, Aalpha, Oomega, Endow, m, n)

obj = lambda x : -pareto_obj(x, llambda_vec, Aalpha, Oomega, Endow, m, n)

x0 = np.ones((m*(n-1),))*0.21
res = minimize(obj, x0, method='BFGS')

Xres_vec = res.x
Xres = Xres_vec.reshape(n-1,m)
Xres_sum = np.sum(Xres, axis=0)
endow_sum = np.sum(Endow, axis=0)
Xres = np.append(Xres, [endow_sum- Xres_sum], axis=0) # pareto optimal allocation
Xres

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

    x_vec0 = np.ones((m,))*0.5
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
    p_vec0 = np.ones((m,))*2.0
    res = minimize(objfct, p_vec0, method='BFGS')
    p_vec1 = res.x
    return p_vec1

# ................
# Symmetric Case I
# ................

Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
Oomega = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5]])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

getEqPrice(Endow, Aalpha, Oomega, m, n)


# ..................
# Asymmetric Case II
# ..................

Aalpha = np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]])
Oomega = np.array([[-0.8, -0.5, -0.2], [-0.5, -0.5, -0.5], [-0.4, -0.5, -0.6]])
Endow = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])

getEqPrice(Endow, Aalpha, Oomega, m, n)



    
