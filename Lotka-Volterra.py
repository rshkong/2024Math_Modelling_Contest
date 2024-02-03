import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

M0 = 100  #雄性数量
F0 = 100  #雌性数量
P0 = 1000  #猎物数量

b_m = 0.07
b_f = 0.03
a = 0.01
r_m=0.1
r_f=0.1
r_p=0.5

def derivative(t,X):
    M,F,P = X
    dMdt = M*(-r_m+b_m*a*P)
    dFdt = F*(-r_f+b_f*a*P)
    dPdt = P*(r_p-a*M-a*F)
    return np.array([dMdt,dFdt,dPdt])


Nt = 1000
tmax = 50
t = np.linspace(0.,tmax,Nt)
X0 = [M0,F0,P0]
res = integrate.solve_ivp(derivative,(0,tmax),X0,t_eval=t)
M,F,P = res.y


plt.figure()
plt.grid()
plt.title("Lotak-Volterra")
plt.plot(t,M,"b",label = "Male")
plt.plot(t,F,'r',label="Female")
plt.plot(t,P,'y',label="Prey")
plt.xlabel('time for 50 generation')
plt.ylabel("population")

plt.show()

