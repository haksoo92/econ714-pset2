import math
from re import S
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import outer
from numpy.lib import polynomial
from numpy.linalg import eig
import scipy
from scipy.sparse import dok
import time
import matplotlib.pyplot as plt
import time
from scipy.optimize import fsolve
import tensorflow as tf


# ======================================================================================================
# Problem 1: Steady State
# Probelm 2: Value Function Iteration with a Fixed Grid 
def runVFI(nk, tolerance):

    t1=time.time()

    aalpha = 1.0/3.0
    bbeta  = 0.97
    ddelta = 0.10

    vProductivity = np.array([-0.05, 0, 0.05],float)
    mTransition   = np.array([[0.97, 0.03, 0.00],
                    [0.01, 0.98, 0.01],
                    [0.00, 0.03, 0.97]],float)

    # 1. Steady State
    klSteadyState = (aalpha/((1/bbeta) - (1-ddelta)))**(1/(aalpha-1))
    qqq1= klSteadyState**aalpha - ddelta*klSteadyState
    qqq2 = (1-aalpha)*(klSteadyState**aalpha)
    lSteadyState = (qqq2/qqq1)**(1/2)
    capitalSteadyState = klSteadyState*lSteadyState
    outputSteadyState = (klSteadyState**aalpha)*lSteadyState
    consumptionSteadyState = lSteadyState*qqq1

    print("Output = ", outputSteadyState, " Capital = ", capitalSteadyState, " Consumption = ", consumptionSteadyState) 


    # 2. VFI with a fixed grid
    def getLabor(cc, kk, zz, aalpha):
        return (np.exp(zz)*(kk**aalpha)/cc)**(1/(1+aalpha))

    vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/nk)
    nGridCapital = len(vGridCapital)
    nGridProductivity = len(vProductivity)

    # mOutput           = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mValueFunction    = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mPolicyFunction   = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    expectedValueFunction = np.zeros((nGridCapital,nGridProductivity),dtype=float)

    # for nProductivity in range(nGridProductivity):
    #     mOutput[:,nProductivity] = np.exp(vProductivity[nProductivity])*(vGridCapital**aalpha)

    maxDifference = 10.0
    # tolerance = 1e-2
    iteration = 0

    while(maxDifference > tolerance):
        expectedValueFunction = np.dot(mValueFunction,mTransition.T)
        for nProductivity in range(nGridProductivity):
            # We start from previous choice (monotonicity of policy function)
            gridCapitalNextPeriod = 0
            for nCapital in range(nGridCapital):
                valueHighSoFar = -100000.0
                capitalChoice  = vGridCapital[0]
                for nCapitalNextPeriod in range(gridCapitalNextPeriod, nGridCapital):
                    kprime = vGridCapital[nCapitalNextPeriod]
                    zz = vProductivity[nProductivity]
                    kk = vGridCapital[nCapital]

                    objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
                    consuption_initialguess = 0.5
                    consumption = fsolve(objfun, consuption_initialguess)
                    labor = getLabor(consumption, kk, zz, aalpha)

                    expectedValueFunction = np.dot(mTransition[nProductivity,:],mValueFunction[nCapitalNextPeriod,:])
                    valueProvisional = (1-bbeta)*(math.log(consumption) - (labor**2)/2) + bbeta*expectedValueFunction
                    if  valueProvisional>valueHighSoFar:
                        valueHighSoFar = valueProvisional
                        capitalChoice = vGridCapital[nCapitalNextPeriod]
                        gridCapitalNextPeriod = nCapitalNextPeriod
                    else:
                        break
                mValueFunctionNew[nCapital,nProductivity] = valueHighSoFar
                mPolicyFunction[nCapital,nProductivity]   = capitalChoice

        maxDifference = (abs(mValueFunctionNew-mValueFunction)).max()

        mValueFunction    = mValueFunctionNew
        mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)

        iteration += 1
        if(iteration%10 == 0 or iteration == 1):
            print(" Iteration = ", iteration, ", Sup Diff = ", maxDifference)
            
    print(" Iteration = ", iteration, ", Sup Duff = ", maxDifference)
    print(" ")
    # print(" My Check = ", mPolicyFunction[1000-1,3-1])
    print(" ")

    t2=time.time()
    print("Elapse time = is ", t2-t1)

    return mValueFunction, mPolicyFunction, t2-t1

# ===========================================================================================
# Problem 3: Accelerator
def runVFI_accelerator(nk, tolerance):

    t1=time.time()

    aalpha = 1.0/3.0
    bbeta  = 0.97
    ddelta = 0.10

    vProductivity = np.array([-0.05, 0, 0.05],float)
    mTransition   = np.array([[0.97, 0.03, 0.00],
                    [0.01, 0.98, 0.01],
                    [0.00, 0.03, 0.97]],float)


    # 1. Steady State
    klSteadyState = (aalpha/((1/bbeta) - (1-ddelta)))**(1/(aalpha-1))
    qqq1= klSteadyState**aalpha - ddelta*klSteadyState
    qqq2 = (1-aalpha)*(klSteadyState**aalpha)
    lSteadyState = (qqq2/qqq1)**(1/2)
    capitalSteadyState = klSteadyState*lSteadyState
    outputSteadyState = (klSteadyState**aalpha)*lSteadyState
    consumptionSteadyState = lSteadyState*qqq1

    print("Output = ", outputSteadyState, " Capital = ", capitalSteadyState, " Consumption = ", consumptionSteadyState) 


    # 2. VFI with a fixed grid
    def getLabor(cc, kk, zz, aalpha):
        return (np.exp(zz)*(kk**aalpha)/cc)**(1/(1+aalpha))

    vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/nk)
    nGridCapital = len(vGridCapital)
    nGridProductivity = len(vProductivity)

    # mOutput           = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mValueFunction    = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    mPolicyFunction   = np.zeros((nGridCapital,nGridProductivity),dtype=float)
    expectedValueFunction = np.zeros((nGridCapital,nGridProductivity),dtype=float)

    # for nProductivity in range(nGridProductivity):
    #     mOutput[:,nProductivity] = np.exp(vProductivity[nProductivity])*(vGridCapital**aalpha)

    maxDifference = 10.0
    # tolerance = 1e-2
    iteration = 0
    iter = 1
    while(maxDifference > tolerance):
        expectedValueFunction = np.dot(mValueFunction,mTransition.T)
        for nProductivity in range(nGridProductivity):
            # We start from previous choice (monotonicity of policy function)
            gridCapitalNextPeriod = 0
            for nCapital in range(nGridCapital):
                valueHighSoFar = -100000.0
                capitalChoice  = vGridCapital[0]
                if iteration % 10 == 0:
                    for nCapitalNextPeriod in range(gridCapitalNextPeriod, nGridCapital):
                        kprime = vGridCapital[nCapitalNextPeriod]
                        zz = vProductivity[nProductivity]
                        kk = vGridCapital[nCapital]

                        objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
                        consuption_initialguess = 0.5
                        consumption = fsolve(objfun, consuption_initialguess)
                        labor = getLabor(consumption, kk, zz, aalpha)

                        expectedValueFunction = np.dot(mTransition[nProductivity,:],mValueFunction[nCapitalNextPeriod,:])
                        valueProvisional = (1-bbeta)*(math.log(consumption) - (labor**2)/2) + bbeta*expectedValueFunction
                        if  valueProvisional>valueHighSoFar:
                            valueHighSoFar = valueProvisional
                            capitalChoice = vGridCapital[nCapitalNextPeriod]
                            gridCapitalNextPeriod = nCapitalNextPeriod
                        else:
                            break
                    mValueFunctionNew[nCapital,nProductivity] = valueHighSoFar
                    mPolicyFunction[nCapital,nProductivity]   = capitalChoice
                else:
                    kprime = mPolicyFunction[nCapital,nProductivity]
                    for ik in range(nGridCapital):
                        if vGridCapital[ik] == kprime:
                            break

                    zz = vProductivity[nProductivity]
                    kk = vGridCapital[nCapital]

                    objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
                    consuption_initialguess = 0.5
                    consumption = fsolve(objfun, consuption_initialguess)
                    labor = getLabor(consumption, kk, zz, aalpha)

                    expectedValueFunction = np.dot(mTransition[nProductivity,:],mValueFunction[ik,:])
                    valueProvisional = (1-bbeta)*(math.log(consumption) - (labor**2)/2) + bbeta*expectedValueFunction

                    mValueFunctionNew[nCapital,nProductivity] = valueProvisional

        maxDifference = (abs(mValueFunctionNew-mValueFunction)).max()

        mValueFunction    = mValueFunctionNew
        mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)

        iteration += 1
        if(iteration%10 == 0 or iteration == 1):
            print(" Iteration = ", iteration, ", Sup Diff = ", maxDifference)
            
    print(" Iteration = ", iteration, ", Sup Duff = ", maxDifference)
    print(" ")

    t2=time.time()
    print("Elapse time = is ", t2-t1)

    return mValueFunction, mPolicyFunction, t2-t1



# RUN VFI
# mValueFunction_vfi, mPolicyFunction_vfi, te_vfi = runVFI(nk = 250, tolerance = 1e-5)
# mValueFunction_acc, mPolicyFunction_acc, te_acc = runVFI_accelerator(nk = 250, tolerance = 1e-5)


nk = 250

aalpha = 1.0/3.0
bbeta  = 0.97
ddelta = 0.10

vProductivity = np.array([-0.05, 0, 0.05],float)
mTransition   = np.array([[0.97, 0.03, 0.00],
                [0.01, 0.98, 0.01],
                [0.00, 0.03, 0.97]],float)

klSteadyState = (aalpha/((1/bbeta) - (1-ddelta)))**(1/(aalpha-1))
qqq1= klSteadyState**aalpha - ddelta*klSteadyState
qqq2 = (1-aalpha)*(klSteadyState**aalpha)
lSteadyState = (qqq2/qqq1)**(1/2)
capitalSteadyState = klSteadyState*lSteadyState
outputSteadyState = (klSteadyState**aalpha)*lSteadyState
consumptionSteadyState = lSteadyState*qqq1

vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/nk)
nGridCapital = len(vGridCapital)
nGridProductivity = len(vProductivity)

# plt.plot(vGridCapital, mPolicyFunction_vfi[:,0], label="Low z")
# plt.plot(vGridCapital, mPolicyFunction_vfi[:,1], label="Medium z")
# plt.plot(vGridCapital, mPolicyFunction_vfi[:,2], label="High z")
# plt.xlabel("k")
# plt.ylabel("k'")
# plt.legend()
# plt.title("Policy Function: VFI over Finite Grid")
# plt.savefig("p2_vfi_policy.png")
# plt.close()

# plt.plot(vGridCapital, mPolicyFunction_acc[:,0], label="Low z")
# plt.plot(vGridCapital, mPolicyFunction_acc[:,1], label="Medium z")
# plt.plot(vGridCapital, mPolicyFunction_acc[:,2], label="High z")
# plt.xlabel("k")
# plt.ylabel("k'")
# plt.legend()
# plt.title("Policy Function: VFI over Finite Grid (accelerator)")
# plt.savefig("p3_vfiacc_policy.png")
# plt.close()

# plt.plot(vGridCapital, mValueFunction_vfi[:,0], label="Low z")
# plt.plot(vGridCapital, mValueFunction_vfi[:,1], label="Medium z")
# plt.plot(vGridCapital, mValueFunction_vfi[:,2], label="High z")
# plt.xlabel("k")
# plt.ylabel("V")
# plt.legend()
# plt.title("Value Function: VFI over Finite Grid")
# plt.savefig("p2_vfi_value.png")
# plt.close()

# plt.plot(vGridCapital, mValueFunction_acc[:,0], label="Low z")
# plt.plot(vGridCapital, mValueFunction_acc[:,1], label="Medium z")
# plt.plot(vGridCapital, mValueFunction_acc[:,2], label="High z")
# plt.xlabel("k")
# plt.ylabel("V")
# plt.legend()
# plt.title("Value Function: VFI over Finite Grid (accelerator)")
# plt.savefig("p3_vfiacc_value.png")
# plt.close()


# ===========================================================================================
# Problem 4: Multigrid
def runVFI_multigrid(tolerance, multigrid = [1e2, 1e3, 1e4]):

    t1=time.time()

    aalpha = 1.0/3.0
    bbeta  = 0.97
    ddelta = 0.10

    vProductivity = np.array([-0.05, 0, 0.05],float)
    mTransition   = np.array([[0.97, 0.03, 0.00],
                    [0.01, 0.98, 0.01],
                    [0.00, 0.03, 0.97]],float)


    # 1. Steady State
    klSteadyState = (aalpha/((1/bbeta) - (1-ddelta)))**(1/(aalpha-1))
    qqq1= klSteadyState**aalpha - ddelta*klSteadyState
    qqq2 = (1-aalpha)*(klSteadyState**aalpha)
    lSteadyState = (qqq2/qqq1)**(1/2)
    capitalSteadyState = klSteadyState*lSteadyState
    outputSteadyState = (klSteadyState**aalpha)*lSteadyState
    consumptionSteadyState = lSteadyState*qqq1

    print("Output = ", outputSteadyState, " Capital = ", capitalSteadyState, " Consumption = ", consumptionSteadyState) 


    # 2. VFI with a fixed grid
    def getLabor(cc, kk, zz, aalpha):
        return ((1-aalpha)*np.exp(zz)*(kk**aalpha)/cc)**(1/(1+aalpha)) # (1-alpha) omitted previously

    isInit = True
    for coarseness in multigrid:
        if isInit:
            isInit = False
            vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/coarseness)
            nGridCapital = len(vGridCapital)
            nGridProductivity = len(vProductivity)
            mValueFunction    = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            mPolicyFunction   = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            expectedValueFunction = np.zeros((nGridCapital,nGridProductivity),dtype=float)
        else:
            prevFineness, _ = mPolicyFunction_saved.shape
            vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/coarseness)
            nGridCapital = len(vGridCapital)
            nGridProductivity = len(vProductivity)
            mValueFunction    = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            mPolicyFunction   = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            expectedValueFunction = np.zeros((nGridCapital,nGridProductivity),dtype=float)
            curridx_nGridCapitalSaved = 0
            for ik in range(nGridCapital):
                if curridx_nGridCapitalSaved >= prevFineness-1:
                    mValueFunction[ik,:] = mValueFunction_saved[prevFineness-1,:]
                    mValueFunctionNew[ik,:] = mValueFunction_saved[prevFineness-1,:]
                    mPolicyFunction[ik,:]   = mPolicyFunction_saved[prevFineness-1,:]
                elif vGridCapital[ik] < vGridCapital_saved[curridx_nGridCapitalSaved+1]:
                    mValueFunction[ik,:] = mValueFunction_saved[curridx_nGridCapitalSaved,:]
                    mValueFunctionNew[ik,:] = mValueFunction_saved[curridx_nGridCapitalSaved,:]
                    mPolicyFunction[ik,:]   = mPolicyFunction_saved[curridx_nGridCapitalSaved,:]
                else:
                    curridx_nGridCapitalSaved += 1
                    mValueFunction[ik,:] = mValueFunction_saved[curridx_nGridCapitalSaved,:]
                    mValueFunctionNew[ik,:] = mValueFunction_saved[curridx_nGridCapitalSaved,:]
                    mPolicyFunction[ik,:]   = mPolicyFunction_saved[curridx_nGridCapitalSaved,:]

        # for nProductivity in range(nGridProductivity):
        #     mOutput[:,nProductivity] = np.exp(vProductivity[nProductivity])*(vGridCapital**aalpha)

        maxDifference = 10.0
        # tolerance = 1e-2
        iteration = 0
        iter = 1
        while(maxDifference > tolerance):
            expectedValueFunction = np.dot(mValueFunction,mTransition.T)
            for nProductivity in range(nGridProductivity):
                # We start from previous choice (monotonicity of policy function)
                gridCapitalNextPeriod = 0
                for nCapital in range(nGridCapital):
                    valueHighSoFar = -100000.0
                    capitalChoice  = vGridCapital[0]
                    if iteration % 10 == 0:
                        for nCapitalNextPeriod in range(gridCapitalNextPeriod, nGridCapital):
                            kprime = vGridCapital[nCapitalNextPeriod]
                            zz = vProductivity[nProductivity]
                            kk = vGridCapital[nCapital]

                            objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
                            consuption_initialguess = 0.5
                            consumption = fsolve(objfun, consuption_initialguess)
                            labor = getLabor(consumption, kk, zz, aalpha)

                            expectedValueFunction = np.dot(mTransition[nProductivity,:],mValueFunction[nCapitalNextPeriod,:])
                            valueProvisional = (1-bbeta)*(math.log(consumption) - (labor**2)/2) + bbeta*expectedValueFunction
                            if  valueProvisional>valueHighSoFar:
                                valueHighSoFar = valueProvisional
                                capitalChoice = vGridCapital[nCapitalNextPeriod]
                                gridCapitalNextPeriod = nCapitalNextPeriod
                            else:
                                break
                        mValueFunctionNew[nCapital,nProductivity] = valueHighSoFar
                        mPolicyFunction[nCapital,nProductivity]   = capitalChoice
                    else:
                        kprime = mPolicyFunction[nCapital,nProductivity]
                        for ik in range(nGridCapital):
                            if vGridCapital[ik] == kprime:
                                break

                        zz = vProductivity[nProductivity]
                        kk = vGridCapital[nCapital]

                        objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
                        consuption_initialguess = 0.5
                        consumption = fsolve(objfun, consuption_initialguess)
                        labor = getLabor(consumption, kk, zz, aalpha)

                        expectedValueFunction = np.dot(mTransition[nProductivity,:],mValueFunction[ik,:])
                        valueProvisional = (1-bbeta)*(math.log(consumption) - (labor**2)/2) + bbeta*expectedValueFunction

                        mValueFunctionNew[nCapital,nProductivity] = valueProvisional

            maxDifference = (abs(mValueFunctionNew-mValueFunction)).max()

            mValueFunction    = mValueFunctionNew
            mValueFunctionNew = np.zeros((nGridCapital,nGridProductivity),dtype=float)

            iteration += 1
            if(iteration%10 == 0 or iteration == 1):
                print(" Iteration = ", iteration, ", Sup Diff = ", maxDifference)
            
            mValueFunction_saved = mValueFunction
            mPolicyFunction_saved = mPolicyFunction
            vGridCapital_saved = vGridCapital
                
        print(" Iteration = ", iteration, ", Sup Duff = ", maxDifference)
        print(" ")
        t2=time.time()
        print("Elapse time = is ", t2-t1)

    return mValueFunction, mPolicyFunction, t2-t1

# RUN
runVFI(nk = 1e3, tolerance = 1e-3)
runVFI_accelerator(nk = 1e3, tolerance = 1e-3)
mValueFunction_mg, mPolicyFunction_mg, te_mg = runVFI_multigrid(tolerance = 1e-3, multigrid=[1e2, 1e3])

vGridCapital = np.arange(0.1*capitalSteadyState,1.9*capitalSteadyState,(1.9*capitalSteadyState-0.1*capitalSteadyState)/1e3)
plt.plot(vGridCapital, mPolicyFunction_mg[:,0], label="Low z")
plt.plot(vGridCapital, mPolicyFunction_mg[:,1], label="Medium z")
plt.plot(vGridCapital, mPolicyFunction_mg[:,2], label="High z")
plt.xlabel("k")
plt.ylabel("k'")
plt.legend()
plt.title("Policy Function: VFI Multigrid")
plt.savefig("p4_vfimg_policy.png")
plt.close()

plt.plot(vGridCapital, mValueFunction_mg[:,0], label="Low z")
plt.plot(vGridCapital, mValueFunction_mg[:,1], label="Medium z")
plt.plot(vGridCapital, mValueFunction_mg[:,2], label="High z")
plt.xlabel("k")
plt.ylabel("V")
plt.legend()
plt.title("Value Function: VFI Multigrid")
plt.savefig("p4_vfimg_value.png")
plt.close()

# ===========================================================================================
# Problem 5: Chebychev
# (1) The Euler equation
# U_c(t) = \beta E_t [U_c(t+1) (\alpha e^{z_{t+1}} k_{t+1}^{\alpha-a} l(k_{t+1}, z_{t+1})^\alpha + (1-\delta) ) ]

# (2) The consumption-labor FOC
# l(k,z) = ((1-\alpha) (1/c) e^z k^\alpha  )^{1/(1+\alpha)}
def getLabor(cc, kk, zz, aalpha):
    return ((1-aalpha)*np.exp(zz)*(kk**aalpha)/cc)**(1/(1+aalpha)) # (1-alpha) omitted previously

def getConsumption(kk, zz, kprime, param):
    aalpha = param["aalpha"]
    ddelta = param["ddelta"]
    objfun = lambda cons: -kprime + np.exp(zz)*(kk**aalpha)*(getLabor(cons, kk, zz, aalpha)**(1-aalpha)) - cons + (1-ddelta)*kk
    consuption_initialguess = 0.5
    consumption = fsolve(objfun, consuption_initialguess)
    return consumption

def kprime(kk, zz, ttheta, param):
        ttheta = np.reshape(ttheta, param["ttheta_shape"])
        idx_zz = (zz == param["vProductivity"])
        currtheta = ttheta[idx_zz,:]
        evalapprox = np.polynomial.chebyshev.chebval(kk, currtheta[0,:])
        if evalapprox <= 0:
            evalapprox = 1e-6
        return evalapprox

def getPsi(k, idx_ki, kfegrid):
    kim1 = kfegrid[idx_ki-1]
    kip1 = kfegrid[idx_ki+1]
    ki = kfegrid[idx_ki]
    if k >= kim1 and k <= ki:
        res = (k-kim1)/(ki-kim1)
    elif k >= ki and k<= kip1:
        res = (kip1-k)/(kip1-ki)
    else:
        res = 0
    return res

def kprime_fem(kk, zz, ttheta, param):
        kfegrid = param["kfegrid"]
        ttheta = np.reshape(ttheta, param["ttheta_shape"])
        idx_zz = (zz == param["vProductivity"])
        currtheta = ttheta[idx_zz,:]

        sum = 0
        for i in range(currtheta.size-2):
            currPsi = getPsi(kk, i+1, kfegrid)
            sum += currtheta[0,i+1] * currPsi

        # if evalapprox <= 0:
        #     evalapprox = 1e-6
        return sum

# nz = 3
# kfegrid = np.array([0.2, 0.25, 1, 1.2, 1.5, 2])
# ttheta = np.zeros((nz, kfegrid.size))
# idx_zz = (zz == param["vProductivity"])
# currtheta = ttheta[idx_zz,:]
# left, right = -1, 0
# for ife in range(kfegrid.size+1):
#     if kk < kfegrid[ife]:
#         break
#     else:
#         if left == kfegrid.size - 1: # last index
#             break
#         else:
#             left += 1
#             right += 1
# if left == -1:
#     val = currtheta[0]
# elif left == kfegrid.size - 1:
#     val = currtheta[left]
# else:
#     val = ((kk-leftval)/(rightval-leftval))*currtheta[left] + ((rightval-kk)/(rightval-leftval))*currtheta[right]

def computeEEErr(kk, zz, ttheta, param):
    aalpha = param["aalpha"]
    ddelta = param["ddelta"]
    vProductivity = param["vProductivity"]
    mTransition = param["mTransition"]

    if param["isFEM"]:
        kkprime = kprime_fem(kk, zz, ttheta, param)
    else:
        kkprime = kprime(kk, zz, ttheta, param)
    cc = getConsumption(kk, zz, kkprime, param)
    lhs = 1/cc
    lhs = lhs[0]

    zzidx = (zz == vProductivity)
    currTransProb = mTransition[zzidx,:]
    currTransProb = currTransProb[0,:]
    rhs = 0
    for i in range(currTransProb.size):
        zzprime = vProductivity[i]
        kprimeprime = kprime(kkprime, zzprime, ttheta, param)
        ccprime = getConsumption(kkprime, zzprime, kprimeprime, param)
        llprime = getLabor(ccprime, kkprime, zzprime, aalpha)
        curr_rhs = (1/ccprime)*(aalpha*np.exp(zzprime)*kkprime**(aalpha-1)*llprime**(aalpha) + (1-ddelta))
        rhs = rhs + currTransProb[i]*curr_rhs[0]
    err = (lhs-rhs)**2
    return err

def sumedEEErr(ttheta, param):
    res = 0 # collocation method
    for iz in range(len(param["vProductivity"])):
        for ik in range(len(param["vGridCapital"])):
            kk = (param["vGridCapital"])[ik]
            zz = (param["vProductivity"])[iz]
            res += computeEEErr(kk, zz, ttheta, param)
    print("Current EEE = ",res)
    return res


# ..................................................................
# SET PARAMETERS
aalpha = 1.0/3.0
bbeta  = 0.97
ddelta = 0.10

vProductivity = np.array([-0.05, 0, 0.05],float)
mTransition   = np.array([[0.97, 0.03, 0.00],
                [0.01, 0.98, 0.01],
                [0.00, 0.03, 0.97]],float)

klSteadyState = (aalpha/((1/bbeta) - (1-ddelta)))**(1/(aalpha-1))
qqq1= klSteadyState**aalpha - ddelta*klSteadyState
qqq2 = (1-aalpha)*(klSteadyState**aalpha)
lSteadyState = (qqq2/qqq1)**(1/2)
capitalSteadyState = klSteadyState*lSteadyState
outputSteadyState = (klSteadyState**aalpha)*lSteadyState

nk = 10
vGridCapital = np.arange(0.7*capitalSteadyState,1.3*capitalSteadyState,(1.7*capitalSteadyState-0.3*capitalSteadyState)/nk)

nz = 3
napprox = 5
ttheta = np.ones((nz,napprox)) * 0.3 # [nz, 5-deg Cheby e.g.]

param = dict()
param["aalpha"] = 1.0/3.0
param["bbeta"]  = 0.97
param["ddelta"] = 0.10
param["vProductivity"] = np.array([-0.05, 0, 0.05],float)
param["mTransition"]   = np.array([[0.97, 0.03, 0.00],
                [0.01, 0.98, 0.01],
                [0.00, 0.03, 0.97]],float)
param["vGridCapital"] = vGridCapital
param["ttheta_shape"] = ttheta.shape
param["tolerance"] = 1e2
param["isFEM"] = False

# while True:
#     # opt = tf.keras.optimizers.Adam(learning_rate=0.1)
#     # ttheta_x = tf.convert_to_tensor(ttheta, np.float32)

#     objfun = lambda ttheta_x: sumedEEErr(ttheta_x, param)
#     # def computeloss(param):
#     #     objfun = sumedEEErr(ttheta_x, param)
#     #     if np.isnan(objfun):
#     #         objfun = np.Inf
#     #     return tf.convert_to_tensor(objfun, np.float32)
#     # step_count = opt.minimize(computeloss(param), var_list=[ttheta_x], tape=tf.GradientTape()).numpy()

#     res = scipy.optimize.minimize(objfun, ttheta, options={"ftol":100})
#     tthetaNew = np.reshape(res.x, ttheta.shape)
#     maxDifference = (abs(tthetaNew-ttheta)).max()
#     print("Max Differenct = ", maxDifference)
#     print("Val = ", tthetaNew)
#     print("Success? ", res['success'])
#     if maxDifference < param["tolerance"]:
#         if res['success']:
#             break
#     else:
#         ttheta = tthetaNew

# ..................................................................
# COMPUTE

# objfun = lambda ttheta_x: sumedEEErr(ttheta_x, param)
# res = scipy.optimize.minimize(objfun, ttheta, options={"ftol":100, "maxiter":10})
# tthetaNew = np.reshape(res.x, ttheta.shape)

tthetaNew = np.array([[ 0.44335047,  0.43895593,  0.27687838, -0.02566881,  0.01528517],
       [ 0.76238755,  0.49826251,  0.05140182,  0.0231563 ,  0.4679174 ],
       [ 0.95213723,  0.48121215, -0.25572105, -0.1886888 ,  0.60561429]])

policy_cheby = np.zeros((vGridCapital.size,3))
for i in range(vGridCapital.size):
    for j in range(3):
        policy_cheby[i,j] = kprime(vGridCapital[i], vProductivity[j], tthetaNew, param)

plt.plot(vGridCapital, policy_cheby[:,0], label="Low z")
plt.plot(vGridCapital, policy_cheby[:,1], label="Medium z")
plt.plot(vGridCapital, policy_cheby[:,2], label="High z")
plt.xlabel("k")
plt.ylabel("k'")
plt.legend()
plt.title("Projection Method using Chebyshev Polynomials")
# plt.show()
plt.close()



# =========================================================================================
# Problem 6: Finite Elements

# param["isFEM"] = True
# param["kfegrid"] = np.array([0.02, 0.1, 0.15, 0.25, 1, 1.2, 1.5, 2])
# ttheta = np.ones((nz,napprox)) * 0.9 # [nz, 5-deg Cheby e.g.]

# objfun = lambda ttheta_x: sumedEEErr(ttheta_x, param)
# res = scipy.optimize.minimize(objfun, ttheta, options={"maxiter":100})
# tthetaNew = np.reshape(res.x, ttheta.shape)

tthetaNew = np.array([[1.27346957, 0.96132841, 0.7050724 , 0.86252981, 0.52249977],
       [1.11692595, 0.84225103, 0.8612759 , 0.8360427 , 0.67221362],
       [1.00541629, 0.70995305, 0.9154824 , 0.87916678, 0.75941391]])

policy_fem = np.zeros((vGridCapital.size,3))
for i in range(vGridCapital.size):
    for j in range(3):
        policy_fem[i,j] = kprime(vGridCapital[i], vProductivity[j], tthetaNew, param)

plt.plot(vGridCapital, policy_fem[:,0], label="Low z")
plt.plot(vGridCapital, policy_fem[:,1], label="Medium z")
plt.plot(vGridCapital, policy_fem[:,2], label="High z")
plt.xlabel("k")
plt.ylabel("k'")
plt.legend()
plt.title("Projection Method using Finite Elements Method")
# plt.show()

