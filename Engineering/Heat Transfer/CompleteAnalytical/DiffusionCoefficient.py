
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

kB = 8.6173324e-5
Temp = np.linspace(500.0,1200.0,num=100)
Dgamma = lambda T: 0.025e-4*np.exp(-6759/(T + 273.15))
Dalpha = lambda T: 1.5e-7*np.exp(-0.088/kB/(T + 273.15))
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(Temp,Dalpha(Temp),label=r'$D_{\alpha}$')
ax.plot(Temp,Dgamma(Temp),label=r'$D_{\gamma}$')
ax.set_xlabel(r'$T$'  + ' ' +  r'$(°C)$')
ax.set_ylabel(r'$D$' + ' ' +  r'$(m^2/s)$')
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='lightgrey', linestyle='--')
plt.minorticks_on()
plt.legend()

df = pd.DataFrame(columns=['T (C)','D_alpha','D_gamma'])
df['T (C)'] = Temp
df['D_alpha'] = Dalpha(Temp)
df['D_gamma'] = Dgamma(Temp)
__location__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
df.to_excel(os.path.join(__location__,'DiffCoef.xlsx'))
plt.show()