import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

M0 = 150  #雄性数量
F0 = 150  #雌性数量
P0 = 500  #猎物数量

b_m = 1.0
s = 0.6
a = 0.01
r_m=1
r_f=1
r_p=2

def derivative(t,X):
    M,F,P = X
    dMdt = -r_m*M+B(M,F,P)*s
    dFdt = -r_f*F+B(M,F,P)*(1-s)
    dPdt = P*(r_p-a*M-a*F)
    return np.array([dMdt,dFdt,dPdt])

def B(M,F,P):
    return b_m*a*P*min([M,F])

Nt = 10000
tmax = 30
t = np.linspace(0.,tmax,Nt)
X0 = [M0,F0,P0]
res = integrate.solve_ivp(derivative,(0,tmax),X0,t_eval=t)
M,F,P = res.y


plt.figure(1)
plt.grid()
plt.title("Lotak-Volterra")
plt.plot(t,M,"b",label = "Male")
plt.plot(t,F,'r',label="Female")
plt.plot(t,P,'y',label="Prey")
plt.xlabel('time for 50 generation')
plt.ylabel("population")
plt.legend()

plt.figure(2)
plt.plot(t,M/(M+F),'r')
plt.title("sex ratio")
plt.show()

