import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm
from scipy.interpolate import splrep, splev
from scipy.interpolate import spline, BSpline
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import Akima1DInterpolator
from scipy.misc import comb
from labellines import labelLine, labelLines
from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import FormatStrFormatter
import copy


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

class CCT:
    search_transformations = ['Ms','Mf','Bs','Bf','Fs','Ff','Ps','Pf']
    lines_kwargs = {
        'Ac1':dict(color='blue'),
        'Ac3':dict(color='red'),
        'Ms':dict(color='green'),
        'Mf':dict(color='darkgreen'),
        'Bs':dict(color='blueviolet'),
        'Bf':dict(color='indigo'),
        'B':dict(color='indigo'),
        'Ps':dict(color='teal'),
        'Pf':dict(color='teal')
    }
    def __init__(self,T0,Ac1,Ac3,transformation_data):
        self.T0 = T0
        self.Ac1 = Ac1
        self.Ac3 = Ac3
        self.transormation_data = transformation_data
        self.treated_data = {}
        self.treat_transformation_data()

    def treat_transformation_data(self):
        for data in self.transormation_data:
            self.treated_data[data['type']] = {'X':[],'Y':[]}

        rates = []
        for data in self.transormation_data:
            self.treated_data[data['type']]['X'].append((data['T0'] - data['T'])/data['rate'])
            self.treated_data[data['type']]['Y'].append(data['T'])
            rates.append(data['rate'])
        self.cooling_rates = list(set(rates))   # Get unique values
        tmp = copy.copy(self.treated_data)
        
        self.whole_curve = {}

        for transformation,transf_data in self.treated_data.items():
            self.treated_data[transformation]['X'] = [x for x,_ in sorted(zip(tmp[transformation]['X'],tmp[transformation]['Y']), key=lambda pair: pair[0])]
            self.treated_data[transformation]['Y'] = [y for _,y in sorted(zip(tmp[transformation]['X'],tmp[transformation]['Y']), key=lambda pair: pair[0])]
        if('Ms' in self.treated_data.keys()):
            self.treated_data['Ms']['X'].insert(0,1e-1)
            self.treated_data['Ms']['Y'].insert(0,self.treated_data['Ms']['Y'][0])
        if('Mf' in self.treated_data.keys()):
            self.treated_data['Mf']['X'].insert(0,1e-1)
            self.treated_data['Mf']['Y'].insert(0,self.treated_data['Mf']['Y'][0])
        if('Bs' in self.treated_data.keys() and 'Bf' in self.treated_data.keys()):  # Bs in counter clockwise
            self.whole_curve['B'] = {'X':list(reversed(self.treated_data['Bs']['X'])) + self.treated_data['Bf']['X'],'Y':list(reversed(self.treated_data['Bs']['Y'])) + self.treated_data['Bf']['Y']}

    def plot_CCT(self,schema=lines_kwargs):
        mode = 'straight'
        fig = plt.figure(figsize=(12,8),dpi=100)
        ax = fig.add_subplot(111)
        time = np.logspace(-1,5,num=2000)
        
        cooling_curves = [self.T0 - i*time for i in self.cooling_rates]
        
        cmap = mpl.cm.get_cmap('Spectral')
        trc_points = [[],[]]
        lines = []
        for k in range(len(cooling_curves)):
            ax.plot(time,cooling_curves[k],'--',color='lightgray',label=str(self.cooling_rates[k]) + ' °C/s')
            pos_x = (self.T0 - 200.0)/self.cooling_rates[k]
            ax.text(pos_x, 200.0 + 5.0 ,str(self.cooling_rates[k]) + ' °C/s', ha='left', va='baseline',fontsize=6,color='gray',zorder=10)
        
        ax.plot(time,self.T0 - 8.33*time,'--',color='black',label=str(8.33) + ' °C/s')
        #pos_x = (self.T0 - 200.0)/8.33
        #ax.text(pos_x, 200.0 + 5.0 ,str(8.33) + ' °C/s', ha='left', va='baseline',fontsize=6,color='black',zorder=10)

        ax.set_xscale('log')
        ax.set_ylim(100.0,900.0)
        ax.set_xlim(1e-1,1e4)
        ax.set_xlabel('t (s)')
        ax.set_ylabel('T (°C)')
        ax.set_title('5Ni')
        #labelLines(plt.gca().get_lines(),zorder=2.5,fontsize=6)
        ax.grid(b=True, which='major', color='gray', linestyle='-')
        ax.grid(b=True, which='minor', color='lightgray', linestyle='-')

        ac3 = plt.hlines(self.Ac3,time[0],time[-1],'r',label='Ac3',zorder=10)
        ac1 = plt.hlines(self.Ac1,time[0],time[-1],'b',label='Ac1',zorder=10)
        ax.text(time[0], self.Ac1, ' ' + r'$Ac_1$' + ' = ' + str(self.Ac1) + ' °C', ha='left', va='bottom',fontsize=10,color=schema['Ac1']['color'])
        ax.text(time[0], self.Ac3, ' ' + r'$Ac_3$' + ' = ' + str(self.Ac3) + ' °C', ha='left', va='bottom',fontsize=10,color=schema['Ac3']['color'])
        if('Ms' in self.treated_data.keys()):
            ax.text(time[0], self.treated_data['Ms']['Y'][0], ' ' + r'$M_s$' + ' = ' + str(self.treated_data['Ms']['Y'][0]) + ' °C', ha='left', va='bottom',fontsize=10,color=schema['Ms']['color'])
            ax.hlines(self.treated_data['Ms']['Y'][0],0.0,1e5,color=schema['Ms']['color'],linestyles='--')
        if('Mf' in self.treated_data.keys()):
            ax.text(time[0], self.treated_data['Mf']['Y'][0], ' ' + r'$M_f$' + ' = ' + str(self.treated_data['Mf']['Y'][0]) + ' °C', ha='left', va='bottom',fontsize=10,color=schema['Mf']['color'])
            ax.hlines(self.treated_data['Mf']['Y'][0],0.0,1e5,color=schema['Mf']['color'],linestyles='--')
        if('Bs' in self.treated_data.keys()):
            ax.text(self.treated_data['Bs']['X'][-1], self.treated_data['Bs']['Y'][-1], ' ' + r'$B_s$', ha='left', va='bottom',fontsize=10,color=schema['Bs']['color'])
        if('Bf' in self.treated_data.keys()):
            ax.text(self.treated_data['Bf']['X'][-1], self.treated_data['Bf']['Y'][-1], ' ' + r'$B_f$', ha='left', va='bottom',fontsize=10,color=schema['Bf']['color'])

        if(mode == 'spline'):
            # Test spline
            order_splines = 1   # Option to smooth curve
            number_of_points = 400

            for transformation,transf_data in self.treated_data.items():
                x_points = self.whole_curve['B']['X']
                log_x_points = np.log10(x_points)
                y_points = self.whole_curve['B']['Y']
                t_points = np.linspace(0.0,1.0,num=len(x_points))
                akima_log_x = Akima1DInterpolator(t_points,log_x_points)
                akima_y = Akima1DInterpolator(t_points,y_points)
                t_new = np.linspace(0.0,1.0,num=1000)
                
                ax.plot(np.power(10.0,akima_log_x(t_new)),akima_y(t_new),label='B',color=schema['B']['color'],zorder=10)

        
            ttx = np.linspace(0.0,1.0,num=len(self.whole_curve['B']['X']))
            tty = np.linspace(0.0,1.0,num=len(self.whole_curve['B']['Y']))
            tt = np.linspace(0.0,1.0,num=number_of_points)
            tx,cx,kx = splrep(ttx,np.log10(self.whole_curve['B']['X']),s=0, k=3)
            ty,cy,ky = splrep(tty,self.whole_curve['B']['Y'],s=0, k=3)
            spline_x = BSpline(tx,cx,kx,extrapolate=False)
            spline_y = BSpline(ty,cy,ky,extrapolate=False)

            
            #ax.plot(np.power(10.0,spline_x(tt)),spline_y(tt),label='B',color='black',zorder=10)
        elif(mode == 'bezier'):
            # Test spline
            order_splines = 3   # Option to smooth curve
            number_of_points = 400
            for transformation,transf_data in self.treated_data.items():
                xnew = np.linspace(min(transf_data['X']),max(transf_data['X']),number_of_points) #300 represents number of points to make between T.min and T.max
                xnew = list(xnew) + transf_data['X']
                xnew = sorted(xnew)
                
                log_xnew = np.log10(xnew)
                log_x = np.log10(transf_data['X'])
                
                points = np.column_stack((transf_data['X'],transf_data['Y']))
                xvals, yvals = bezier_curve(points, nTimes=100)
                #y_smooth = spline(log_x,transf_data['Y'],log_xnew,order=order_splines)

                ax.plot(xvals,yvals,label=transformation,color=schema[transformation]['color'],zorder=10)
                ax.scatter(transf_data['X'],transf_data['Y'],label=None,color=schema[transformation]['color'],zorder=10,marker='.',s=10)

        elif(mode == 'straight'):
            for transformation,transf_data in self.treated_data.items():
                #ax.plot(transf_data['X'],transf_data['Y'],label=transformation,color=schema[transformation]['color'],zorder=10)
                ax.scatter(transf_data['X'],transf_data['Y'],label=None,color=schema[transformation]['color'],zorder=10,marker='.',s=10)
            
            transformations = ['B','Ms','Mf']
            for transf in transformations:
                if transf in self.whole_curve.keys():
                    x_points = self.whole_curve[transf]['X']
                    y_points = self.whole_curve[transf]['Y']
                else:
                    x_points = self.treated_data[transf]['X']
                    y_points = self.treated_data[transf]['Y']
                log_x_points = np.log10(x_points)
                
                t_points = np.linspace(0.0,1.0,num=len(x_points))
                akima_log_x = Akima1DInterpolator(t_points,log_x_points)
                akima_y = Akima1DInterpolator(t_points,y_points)
                t_new = np.linspace(0.0,1.0,num=1000)
                
                ax.plot(np.power(10.0,akima_log_x(t_new)),akima_y(t_new),label=transf,color=schema[transf]['color'],zorder=10)
            #points = np.column_stack((np.log10(self.whole_curve['B']['X']),self.whole_curve['B']['Y']))
            #xvals, yvals = bezier_curve(points, nTimes=1000)
            #y_smooth = spline(log_x,transf_data['Y'],log_xnew,order=order_splines)
            #ax.plot(np.power(10.0,xvals),yvals,label='B',color='black',zorder=10)

            #ax.scatter(transf_data['X'],transf_data['Y'],label=None,color=schema[transformation]['color'],zorder=10,marker='.',s=10)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        
if __name__=='__main__':

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    '''
    ac1 = 837.9
    ac3 = 885.3
    transormation_data_TENAX300 = [
        dict(type='Ms',T0=1020.0,rate=50.0,T=329.0),
        dict(type='Ms',T0=1020.0,rate=2.0,T=340.0),
        dict(type='Ms',T0=1020.0,rate=1.0,T=336.1),
        dict(type='Ms',T0=1020.0,rate=0.5,T=336.4),
        dict(type='Ms',T0=1020.0,rate=0.2,T=268.0),
        dict(type='Ms',T0=1020.0,rate=0.1,T=274.0),
        dict(type='Mf',T0=1020.0,rate=50.0,T=120.0),
        dict(type='Mf',T0=1020.0,rate=2.0,T=70.6),
        dict(type='Mf',T0=1020.0,rate=1.0,T=120.0),
        dict(type='Mf',T0=1020.0,rate=0.5,T=172.2),
        dict(type='Mf',T0=1020.0,rate=0.2,T=101.0),
        dict(type='Mf',T0=1020.0,rate=0.1,T=147.2),

        dict(type='Bs',T0=1020.0,rate=0.5,T=336.4),
        dict(type='Bs',T0=1020.0,rate=0.2,T=362.4),
        dict(type='Bs',T0=1020.0,rate=0.1,T=394.4),
        dict(type='Bs',T0=1020.0,rate=0.05,T=400.0),
        dict(type='Bs',T0=1020.0,rate=0.03,T=425.9),

        dict(type='Bf',T0=1020.0,rate=0.1,T=274.0),
        dict(type='Bf',T0=1020.0,rate=0.05,T=246.7),
        dict(type='Bf',T0=1020.0,rate=0.03,T=260.9),

        dict(type='Ps',T0=1020.0,rate=0.06,T=766.2),
        dict(type='Ps',T0=1020.0,rate=0.05,T=790.0),
        dict(type='Ps',T0=1020.0,rate=0.03,T=827.7),

        dict(type='Pf',T0=1020.0,rate=0.06,T=766.2),
        dict(type='Pf',T0=1020.0,rate=0.05,T=750.0),
        dict(type='Pf',T0=1020.0,rate=0.03,T=704.7),
    ]
    '''
    ac1 = 646.9
    ac3 = 793.0
    transormation_data_5Ni_long = [
        dict(type='Ms',T0=885.0,rate=100.0,T=445.0),
        dict(type='Mf',T0=885.0,rate=100.0,T=243.0),
        dict(type='Ms',T0=885.0,rate=50.0,T=446.0),
        dict(type='Mf',T0=885.0,rate=50.0,T=286.0),
    ]

    transormation_data_5Ni_transv = [
        dict(type='Ms',T0=885.0,rate=100.0,T=446.0),
        dict(type='Mf',T0=885.0,rate=100.0,T=234.0),
        dict(type='Ms',T0=885.0,rate=50.0,T=446.0),
        #dict(type='Bs',T0=885.0,rate=50.0,T=573.0),
        #dict(type='Bf',T0=885.0,rate=50.0,T=573.0),
        dict(type='Mf',T0=885.0,rate=50.0,T=294.0),
        dict(type='Bs',T0=885.0,rate=10.0,T=578.0),
        dict(type='Bf',T0=885.0,rate=10.0,T=430.0),
        dict(type='Ms',T0=885.0,rate=10.0,T=392.0),
        dict(type='Mf',T0=885.0,rate=10.0,T=293.0),
        dict(type='Bs',T0=885.0,rate=1.0,T=596.0),
        dict(type='Bf',T0=885.0,rate=1.0,T=407.0),
        dict(type='Bs',T0=885.0,rate=0.1,T=599.0),
        dict(type='Bf',T0=885.0,rate=0.1,T=405.0),
    ]

       
    c = CCT(885.0,ac1,ac3,transormation_data_5Ni_transv)
    c.plot_CCT()
    plt.show()
