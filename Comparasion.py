import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

import parameters



M0 = parameters.M0  #雄性数量
F0 = parameters.F0  #雌性数量
P0 = parameters.P0  #猎物数量

b_m = parameters.b_m
s = parameters.s
a = parameters.a
r_m=parameters.r_m
r_f=parameters.r_f
r_p=parameters.r_p
alpha = parameters.alpha

#S函数参数
S_a,S_b=[parameters.S_a,parameters.S_b]
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


def moving_average(x,w):
    temp=np.convolve(x, np.ones(w), 'valid') / w
    res=np.concatenate((temp,temp[-w:-1]))
    return res


Nt = 10000
tmax = 30
t = np.linspace(0.,tmax,Nt)
X0 = [M0,F0,P0]
res = integrate.solve_ivp(derivative,(0,tmax),X0,t_eval=t)
M,F,P = res.y

res_nosex = integrate.solve_ivp(deri_nosex,(0,tmax),X0,t_eval=t)
M_nosex,F_nosex,P_nosex = res_nosex.y


plt.figure(1)
plt.grid()
plt.title("Lotka-Volterra",fontsize=15)
# plt.plot(t,M,"b",label = "Male")
# plt.plot(t,F,'r',label="Female")
plt.plot(t,M+F,"mediumslateblue",label = "Lamprey population")
plt.plot(t,P,'y',label="Prey")
# plt.plot(t,moving_average(M,2000),"b--",label="Male moving average")
plt.plot(t,moving_average(M+F,2000),"mediumslateblue",linestyle="--",label="Population moving average")
plt.plot(t,moving_average(P,2000),"y--",label="Prey moving average")
plt.plot
plt.xlabel('time',fontsize=15)
plt.ylabel("population",fontsize=15)
plt.legend()

# plt.figure(2)
# plt.grid()
# plt.plot(t,M/(M+F),'r')
# plt.title("sex ratio",fontsize=15)

plt.figure(3)
plt.grid()
plt.title("Lotka-Volterra with 1:1 sex ratio",fontsize=15)
plt.plot(t,M_nosex*2,"mediumslateblue",label = "Lamprey population")
# plt.plot(t,M_nosex,'r',label="Female")
plt.plot(t,P_nosex,'y',label="Prey")
plt.plot(t,moving_average(M*2,2000),"mediumslateblue",linestyle="--",label="Population moving average")
plt.plot(t,moving_average(P,2000),"y--",label="Prey moving average")
plt.plot
plt.xlabel('time',fontsize=15)
plt.ylabel("population",fontsize=15)
plt.legend()



plt.show()

print("可变性别比例模型种群数量峰值:")
