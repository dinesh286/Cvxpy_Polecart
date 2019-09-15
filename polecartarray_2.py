import numpy as np
import scipy as sp
import cvxpy as cvx
import matplotlib.pyplot as plt



g = 9.8
l = 1.0
dt = 0.05
K = 400



def f(x, u):
    #print(x)
    b = np.zeros_like(x)
    theta = x[0]
    dtheta = x[1]
    a = u[0]
    b[0] = dtheta
    b[1] = (a * np.cos(theta) - g * np.sin(theta)) / l
    return b

def df(x, u):
    A = np.zeros((x.shape[0], x.shape[0]))
    B = np.zeros((x.shape[0], u.shape[0]))
    theta = x[0]
    dtheta = x[1]
    a = u[0]
    # dthetadot / dtheta
    A[0,1] = 1
    # dtheta derviatvie.
    A[1,0] = (- a * np.sin(theta) - g * np.cos(theta)) / l
    B[1,0] = np.cos(theta) / l
    #b[1,:] = (a * np.cos(theta) - g * np.sin(theta)) / l
    return A, B

def linf(x, u, x2, u2):
    b = f(x,u)
    A, B = df(x,u)
    return b + A*(x2 - x)+B*(u2-u)



np_x = np.zeros((K+1, 2))
np_u = np.zeros((K+1, 1))



for j in range(400):
    x=cvx.Variable((K+1,2))
    u=cvx.Variable((K+1,1)) 

    const =[]
    cost = 0


    for k in range (K):
        if k>0:
            const +=[u[k,:]<= 8]
            const +=[u[k,:]>=-8]    
        else:
            const +=[u[0,:]<= 13]
            const +=[u[0,:]>=-13]                

    const+=[x[0,:] == 0]
    const +=[u[-1,:]<= 8]
    const +=[u[-1,:]>=-8]





#const+=[x[1] == 0]


    for k in range(K):

        const+= [x[2*k+1,:]            == x[2*k+2,:]/2 + x[2*k,:]/2 + dt/8*(linf(np_x[2*k,:], np_u[k,:],x[2*k,:], u[k,:] ) - linf(np_x[2*k+2,:], np_u[k+1,:],x[2*k+2,:], u[k+1,:]))]
        const+= [x[2*k+2,:] - x[2*k,:] ==  (linf(np_x[2*k,:], np_u[k,:], x[2*k,:], u[k,:] ) + 4 * linf(np_x[2*k+1,:], (np_u[k,:] + np_u[k+1,:]) / 2, x[2*k+1], (u[k,:] + u[k+1,:])/2) + linf(np_x[2*k+2,:], np_u[k+1,:], x[2*k+2], u[k,:]))*dt/6]   


        #const += [x[k+1,:] - x[k,:] ==   0.5*dt*(linf(np_x[k+1,:],np_u[k+1,:],x[k+1,:],u[k+1,:])+linf(np_x[k,:],np_u[k,:],x[k,:],u[k,:]))]


        cost = cost + 0.5*cvx.huber( x[k,0] - np.pi, M=0.5) + 0.01 * cvx.huber(u[k,:]) +  0.5*cvx.huber( x[k,1] - 0, M=0.25)
       

    cost = cost + 100*cvx.square(x[K,0]-np.pi)+100*cvx.square(x[K,1]-0)
    objective = cvx.Minimize(cost)
    print("Iteration",j+1)
    prob =  cvx.Problem(objective,const)
    sol = prob.solve(verbose=True)
    np_x = x.value
    np_u = u.value
    #print((np_x))
#    print(np.shape(np_u))    

plt.plot(np_x[:,0])
plt.plot(np_x[:,1])
plt.plot(np_u[:,0])

plt.show()





