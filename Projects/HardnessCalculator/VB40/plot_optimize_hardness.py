from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib import cm
from hardnessmodel import *
from HB_Data import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

files = []
files.append(open(os.path.join(__location__,"c82 HB.dat"),'r'))

l = []
hb_data = []

k = 0
for archive in files:
    hb_data.append(HB_Data())
    l.append([])
    for line in archive:
        l[len(l)- 1].append(line.split())
    hb_data[len(hb_data) - 1].readData(l[len(l) - 1])


T_min_model, T_max_model = [430.0,760.0]
H0_model = 500.0
H_far_model = 200.0
model = HardnessModel()
model.change_parameters(H0_model,H_far_model,T_min_model,T_max_model)

def f(X, m, D0, QD,Qinf):
    t,T = X
    return model.functional(m,D0,QD,Qinf).tau(t,T)
    #return model.functional(m,D0,QD,200.0).tau(t,T)

ln_t_min, ln_t_max = [0.5, 6.0]
T_min, T_max = [500.0, 700.0]
n_t, n_T = [10,10]
Nsurf_t, Nsurf_T = [20,20]
a0 = [0.3, 0.9, 500.0, 2000.0]


noise_amplitude = 0.05

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


#t = np.logspace(ln_t_min, ln_t_max, num=n_t, base=2.0)
#T = np.linspace(T_min,T_max,n_T)
#mesh = np.meshgrid(t,T,sparse=False,indexing='ij')
#tv, Tv = mesh
#tt = tv.flatten()
#TT = Tv.flatten()

TT = []
tt = []
log2_tt = []
T_t = []
hard = []
log_hard = []


#z = f((tt,TT), a_true[0], a_true[1], a_true[2], a_true[3]) + noise_amplitude*np.random.randn(len(t)*len(T))
for Temp in hb_data[0].data:
    for time in Temp[1]:
        for h in time[1]:
            T_t.append((time[0],Temp[0]))
            TT.append(float(Temp[0]))
            tt.append(float(time[0]))
            log2_tt.append(log(float(time[0]),2))
            hard.append(model.ConvertToTau(float(h)))
            log_hard.append(np.log(model.ConvertToTau(float(h))))


kwarg = [('loss', 'cauchy'),('f_scale',0.1)]
p, conv = curve_fit(f, (tt,TT), hard, a0, method='lm')
plt.hold(True)
p_sigma = np.sqrt(np.diag(conv))

x_surf, y_surf = np.meshgrid(np.logspace(ln_t_min, ln_t_max, num=Nsurf_t, base=2.0),np.linspace(T_min,T_max,Nsurf_T), sparse=False,indexing='ij')
z_surf = np.log(f((x_surf,y_surf), p[0],p[1], p[2], p[3]))
z_surf_low = np.log(f((x_surf,y_surf), p[0] , p[1], p[2], p[3]))
z_surf_high = np.log(f((x_surf,y_surf), p[0] , p[1], p[2], p[3]))
ax.plot_surface(x_surf,y_surf,z_surf, cmap=cm.inferno,alpha=0.7)
ax.plot_surface(x_surf,y_surf,z_surf_low, cmap=cm.cool,alpha=0.1)
ax.plot_surface(x_surf,y_surf,z_surf_high, cmap=cm.hot,alpha=0.1)
ax.scatter(tt,TT,log_hard)


#ax.xaxis._set_scale('log')
ax.set_xlabel('t')
ax.set_ylabel('T')
ax.set_zlabel('Tau')

print(p)
print(p_sigma)
#plt.plot(x,map(lambda t,T: f(result.x,t,T),x,y))
plt.show()
