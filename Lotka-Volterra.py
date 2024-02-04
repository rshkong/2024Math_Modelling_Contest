import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

M0 = 150  #雄性数量
F0 = 150  #雌性数量
P0 = 500  #猎物数量

b_m = 1
s = 0.6
a = 0.01
r_m=1
r_f=1
r_p=2
alpha = 0.001

#S函数参数
S_a,S_b=[0.0055,-553]
def S(P):
    '''sex ratio'''
    return 11/(50*(1+np.exp(S_a*(P+S_b))))+0.56


def derivative(t,X):
    M,F,P = X
    s=S(P)
    dMdt = -r_m*M+B(M,F,P)*S(P)
    dFdt = -r_f*F+B(M,F,P)*(1-S(P))
    dPdt = P*(r_p-a*M-a*F)#-alpha*P**2
    return np.array([dMdt,dFdt,dPdt])


def deri_nosex(t,X):
    M,F,P = X
    s=S(P)
    dMdt = -r_m*M+B(M,F,P)*0.5
    dFdt = -r_f*F+B(M,F,P)*0.5
    dPdt = P*(r_p-a*M-a*F)#-alpha*P**2
    return np.array([dMdt,dFdt,dPdt])


def B(M,F,P):
    return b_m*a*P*min([M,F])

Nt = 10000
tmax = 30
t = np.linspace(0.,tmax,Nt)
X0 = [M0,F0,P0]
res = integrate.solve_ivp(derivative,(0,tmax),X0,t_eval=t)
M,F,P = res.y

def moving_average(x,w):
    temp=np.convolve(x, np.ones(w), 'valid') / w
    res=np.concatenate((temp,temp[-w:-1]))
    return res


plt.figure(1)
plt.grid()
plt.title("Lotka-Volterra",fontsize=15)
plt.plot(t,M,"b",label = "Male")
plt.plot(t,F,'r',label="Female")
plt.plot(t,P,'y',label="Prey")
plt.plot(t,moving_average(M,2000),"b--",label="Male moving average")
plt.plot(t,moving_average(P,2000),"y--",label="Prey moving average")
plt.plot
plt.xlabel('time',fontsize=15)
plt.ylabel("population",fontsize=15)
plt.legend()

plt.figure(2)
plt.grid()
plt.plot(t,M/(M+F),'r')
plt.title("sex ratio",fontsize=15)
plt.show()
