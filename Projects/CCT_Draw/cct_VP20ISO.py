
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
#from scipy.misc import comb
from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import copy

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

class VillaresColors:
    def __init__(self):
        self.brand_colors = {}
        self.brand_colors['Pantone 294 CVC'] = {
            'Name':'C100M56Y0K18',
            'Rank':1,
            'H':145,
            'S':255,
            'L':77,
            'R':0,
            'G':90,
            'B':155,
            'Code':'005A9B'
        }
        self.brand_colors['Pantone 294 CVC 1'] = {
            'Name':'C100M56Y0K18',
            'Rank':1,
            'H':145,
            'S':255,
            'L':77,
            'R':0,
            'G':92,
            'B':209,
            'Code':'005A9B'
        }

        self.brand_colors['Pantone 294 CVC 2'] = {
            'Name':'C100M56Y0K18',
            'Rank':1,
            'H':145,
            'S':255,
            'L':77,
            'R':0,
            'G':85,
            'B':151,
            'Code':'005A9B'
        }
        self.brand_colors['Pantone Cool Gray 8C'] = {
            'Name':'Pantone Cool Gray 8C',
            'Rank':2,
            'H':159,
            'S':5,
            'L':110,
            'R':108,
            'G':109,
            'B':112,
            'Code':'6C6D70'
        }
        self.brand_colors['Pantone Cool Gray 8C 1'] = {
            'Name':'Pantone Cool Gray 8C',
            'Rank':2,
            'H':159,
            'S':5,
            'L':110,
            'R':77,
            'G':77,
            'B':77,
            'Code':'6C6D70'
        }
        self.brand_colors['Pantone Process Black'] = {
            'Name':'C0M0Y0K100',
            'Rank':3,
            'H':244,
            'S':16,
            'L':32,
            'R':34,
            'G':30,
            'B':31,
            'Code':'221E1F'
        }
        self.ranked_colors = [self.brand_colors['Pantone 294 CVC 2'],self.brand_colors['Pantone Cool Gray 8C'],self.brand_colors['Pantone Process Black']]


class CCT:
    
    def __init__(self,T0,Ac1,Ac3,transformation_data):
        self.transformation_data = transformation_data
        self.T0 = T0
        self.Ac1 = Ac1
        self.Ac3 = Ac3
        self.treated_data = {}
        #self.cooling_rates = [50.0,2.0,0.5,0.3,0.1,0.05,0.03,0.02,0.01]
        #self.cooling_rates = [100.0/2**k for k in range(0,12)]
        self.treat_transformation_data()


    def treat_transformation_data(self):
        for data in self.transformation_data:
            self.treated_data[data['type']] = {'X':[],'Y':[],'R':[]}

        rates = []
        for data in self.transformation_data:
            self.treated_data[data['type']]['X'].append((data['T0'] - data['T'])/data['rate'])
            self.treated_data[data['type']]['Y'].append(data['T'])
            self.treated_data[data['type']]['R'].append(data['rate'])
            rates.append(data['rate'])
        self.cooling_rates = list(set(rates))   # Get unique values
        tmp = copy.copy(self.treated_data)
        
        self.whole_curve = {}

        for transformation,transf_data in self.treated_data.items():
            self.treated_data[transformation]['X'] = [x for x,_ in sorted(zip(tmp[transformation]['X'],tmp[transformation]['Y']), key=lambda pair: pair[0])]
            self.treated_data[transformation]['Y'] = [y for _,y in sorted(zip(tmp[transformation]['X'],tmp[transformation]['Y']), key=lambda pair: pair[0])]
        '''
        if('Ms' in self.treated_data.keys()):
            self.treated_data['Ms']['X'].insert(0,1e-1)
            self.treated_data['Ms']['Y'].insert(0,self.treated_data['Ms']['Y'][0])
        if('Mf' in self.treated_data.keys()):
            self.treated_data['Mf']['X'].insert(0,1e-1)
            self.treated_data['Mf']['Y'].insert(0,self.treated_data['Mf']['Y'][0])
        if('Bs' in self.treated_data.keys() and 'Bf' in self.treated_data.keys()):  # Bs in counter clockwise
            self.whole_curve['B'] = {'X':list(reversed(self.treated_data['Bs']['X'])) + self.treated_data['Bf']['X'],'Y':list(reversed(self.treated_data['Bs']['Y'])) + self.treated_data['Bf']['Y']}
        '''

    def plot_CCT(self,color_schema,font_schema,ticks_schema,grid_schema,**kwargs):
        name = 'cct_VP20ISO'
        colors = [(c['R']/255.0,c['G']/255.0,c['B']/255.0) for c in color_schema]
        lines_kwargs = {
            'Ac1':dict(color=colors[2]),
            'Ac3':dict(color=colors[2]),
            'T0':dict(color=colors[2]),
            'Ms':dict(color=colors[1]),
            'Mf':dict(color=colors[1]),
            'Bs':dict(color=colors[0]),
            'Bf':dict(color=colors[0]),
            'Bi':dict(color=colors[0]),
            'B':dict(color='indigo'),
            'Ps':dict(color='teal'),
            'Pf':dict(color='teal')
        }
        
        schema=lines_kwargs
        image_formats = kwargs.get('image_formats',['png','svg'])
        show = kwargs.get('show',True)
        dpi_choice = kwargs.get('dpi_choice',400)
        norm = kwargs.get('norm',False)
        
        lines_labes_fontschema = copy.copy(font_schema)
        lines_labes_fontschema['fontsize'] = 12

        fig = plt.figure(figsize=(12,8),dpi=100)
        ax = fig.add_subplot(111)
        
        
        lw_cooling_rates = 1.2


        language = 'English'
        language_data = dict(
            English={'Xlabel':'t (s)','Ylabel':'T ' + r'$\mathrm{(°C)}$','Y2label':'T ' + r'$\mathrm{(°F)}$'},
        )
        plot_limits = {'X':[1e0,1e5],'Y':[0.0,1000.0]}

        lines = []
        Tref = 400.0
        ax.set_xscale('log')
        #self.cooling_rates = [100.0/2**k for k in range(0,15)]
        self.cooling_rates = [100.0, 50.0, 25.0, 12.5, 6.0, 3.0, 1.5, 0.75, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
        sorted_cooling_rates = sorted(self.cooling_rates)
        sorted_cooling_rates.reverse()
        time = np.logspace(-1,5,num=10000)
        cooling_curves = [self.T0 - i*time for i in sorted_cooling_rates]
        ratio_start_curve = 0.995*self.T0
        for k in range(len(sorted_cooling_rates)):
            start_point_index = 0
            if(k != len(sorted_cooling_rates) - 1):
                for j in range(len(time)):
                    if(cooling_curves[k][j] < ratio_start_curve):
                        start_point_index = j
                        break
            print(start_point_index)      
            ax.plot(time[start_point_index:],np.array(cooling_curves[k])[start_point_index:],'--',label=str(sorted_cooling_rates[k]) + ' °C/s',color='gray',alpha=0.7,lw=lw_cooling_rates)
            pos_y = Tref + 7.0 + 20.0*k
            pos_x = (self.T0 - pos_y + 20.0)/(sorted_cooling_rates[k])
            if(pos_x >= plot_limits['X'][0] and pos_x < plot_limits['X'][-1]):
                ax.text(pos_x, pos_y,str(sorted_cooling_rates[k]) + ' °C/s', ha='left', va='baseline',fontsize=8,color='gray',zorder=10)

        
        

        ac3 = plt.hlines(self.Ac3,time[0],time[-1],schema['Ac3']['color'],label=r'$Ac_3$',zorder=10,lw=2.0)
        ac1 = plt.hlines(self.Ac1,time[0],time[-1],schema['Ac1']['color'],label=r'$Ac_1$',zorder=10,lw=2.0)
        Line_T0 = plt.hlines(self.T0,plot_limits['X'][0],plot_limits['X'][-1],schema['Ac1']['color'],linestyle='--',label='T0',zorder=10,lw=1.5)
        ax.text(time[0], self.Ac1, ' ' + r'$Ac_1$' + ' = ' + str(self.Ac1) + ' °C', ha='left', va='bottom',color=schema['Ac1']['color'],**lines_labes_fontschema)
        ax.text(time[0], self.Ac3, ' ' + r'$Ac_3$' + ' = ' + str(self.Ac3) + ' °C', ha='left', va='bottom',color=schema['Ac3']['color'],**lines_labes_fontschema)

        if('Ms' in self.treated_data.keys()):
            ax.text(plot_limits['X'][0], self.treated_data['Ms']['Y'][0], ' ' + r'$M_s$' + ' = ' + str(self.treated_data['Ms']['Y'][0]) + ' °C', ha='left', va='bottom',color=schema['Ms']['color'],**lines_labes_fontschema)
        if('Mf' in self.treated_data.keys()):
            ax.text(plot_limits['X'][0], self.treated_data['Mf']['Y'][0], ' ' + r'$M_f$' + ' = ' + str(self.treated_data['Mf']['Y'][0]) + ' °C', ha='left', va='bottom',color=schema['Mf']['color'],**lines_labes_fontschema)
        if('Bs' in self.treated_data.keys()):
            ax.text(self.treated_data['Bs']['X'][-1] , self.treated_data['Bs']['Y'][-1] , ' ' + r'$B_s$', ha='left', va='bottom',color=schema['Bs']['color'],**lines_labes_fontschema)
        if('Bf' in self.treated_data.keys()):
            ax.text(self.treated_data['Bf']['X'][-1], self.treated_data['Bf']['Y'][-1], ' ' + r'$B_f$', ha='left', va='bottom',color=schema['Bf']['color'],**lines_labes_fontschema)
        if('Ps' in self.treated_data.keys()):
            ax.text(self.treated_data['Ps']['X'][-1] , self.treated_data['Ps']['Y'][-1] , ' ' + r'$P_s$', ha='left', va='bottom',color=schema['Ps']['color'],**lines_labes_fontschema)
        if('Pf' in self.treated_data.keys()):
            ax.text(self.treated_data['Pf']['X'][-1], self.treated_data['Pf']['Y'][-1], ' ' + r'$P_f$', ha='left', va='bottom',color=schema['Pf']['color'],**lines_labes_fontschema)
        if('C' in self.treated_data.keys()):
            ax.text(self.treated_data['C']['X'][-1], self.treated_data['C']['Y'][-1], ' ' + r'$C$', ha='left', va='bottom',color=schema['C']['color'],**lines_labes_fontschema)
        if('Bi' in self.treated_data.keys()):
            ax.text(self.treated_data['Bi']['X'][-1], self.treated_data['Bi']['Y'][-1], ' ' + r'$B_i$', ha='left', va='bottom',color=schema['Bi']['color'],**lines_labes_fontschema)

        transformations = ['Bs','Bf','Bi','Ms','Mf']
        for transf in transformations:
            if transf in self.treated_data.keys():
                x_points = self.treated_data[transf]['X']
                y_points = self.treated_data[transf]['Y']
                log_x_points = np.log10(x_points)
            
                t_points = np.linspace(0.0,1.0,num=len(x_points))
                akima_log_x = Akima1DInterpolator(t_points,log_x_points)
                akima_y = Akima1DInterpolator(t_points,y_points)
                t_new = np.linspace(0.0,1.0,num=100000)
                
                ax.plot(np.power(10.0,akima_log_x(t_new)),akima_y(t_new),label=transf,color=schema[transf]['color'],zorder=10,lw=2.0)


                if(transf == 'Ms'):
                    plt.hlines(self.treated_data['Ms']['Y'][0],1e-1,self.treated_data['Ms']['X'][0],schema[transf]['color'],zorder=10,lw=2.0)
                elif(transf == 'Mf'):
                    plt.hlines(self.treated_data['Mf']['Y'][0],1e-1,self.treated_data['Mf']['X'][0],schema[transf]['color'],zorder=10,lw=2.0)
                

        
        ax.grid(which='minor',**grid_schema['minor'],axis='x')
        ax.grid(which='major',**grid_schema['major'])
        ax.set_axisbelow(True)
        #plt.figtext(0.53, 0.005, lower_text[language], wrap=True, horizontalalignment='center', fontproperties=font_schema['fontproperties'],fontsize=12)
        
        #ax.set_title('VP20ISO',**font_schema)
        ax.set_xlabel(language_data[language]['Xlabel'],**font_schema)
        ax.set_ylabel(language_data[language]['Ylabel'],**font_schema)
        ax.set_xlim(plot_limits['X'][0],plot_limits['X'][1])
        ax.set_ylim(plot_limits['Y'][0],plot_limits['Y'][1])


        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) :    
            label.set_fontproperties(ticks_schema['fontproperties'])
            label.set_fontsize(ticks_schema['fontsize'])
        #plt.legend()
        
        plt.minorticks_on()
        ax2 = ax.twinx()
        ax2.set_yticks(ax.get_yticks())
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel(language_data[language]['Y2label'],**font_schema)
        labels = [i for i in ax2.get_yticks()]
        labels_new = [int(float(i)/5.0*9.0 + 32.0) for i in labels]
        ax2.set_yticklabels(labels_new)
        for label in list(ax2.get_yticklabels()):    
            label.set_fontproperties(ticks_schema['fontproperties'])
            label.set_fontsize(ticks_schema['fontsize'])
            print(label)

        plt.tight_layout()
        loc = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
        [plt.savefig(os.path.join(loc,name + '_' + language + '.' + k),format=k,dpi=dpi_choice) for k in image_formats]
        if(show):
            plt.show()
        plt.close()

if __name__=='__main__':
    fpath = os.path.join(rcParams["datapath"], r"fonts\ttf\FuturaStd-Book.ttf")
    fp = matplotlib.font_manager.FontProperties(fname=fpath)
    colors = VillaresColors().ranked_colors
    font_kwargs = dict(fontproperties=fp,fontsize=16)
    ticks_kwargs = dict(fontproperties=fp,fontsize=14)

    first_version_grid = dict(
        major=dict(ls='-',color='gray'),
        minor=dict(ls=':',color='gainsboro'),
        bar=dict(which='major',b=True,ls=':',color='gray')
        )
    second_version_grid = dict(
        major=dict(b=True,lw=0.7,ls='-',color='gray'),
        minor=dict(b=False,lw=0.7,ls='-',color='gray',alpha=0.3),
        bar=dict(which='major',b=True,lw=0.2,ls=':',color='gray')
        )
    grid_kwargs = second_version_grid

    T0 = 900.0
    Ac1 = 706.4
    Ac3 = 778.0
    transormation_data = [
        dict(type='Ms',T0=T0,rate=50.0,T=330.0),
        dict(type='Ms',T0=T0,rate=1.0,T=330.0),
        dict(type='Ms',T0=T0,rate=0.2,T=330.0),
        dict(type='Bs',T0=T0,rate=0.1,T=350.0),
        dict(type='Bs',T0=T0,rate=0.05,T=420.0),
        dict(type='Bs',T0=T0,rate=0.02,T=420.0),
        dict(type='Bs',T0=T0,rate=0.01,T=438.0),
        dict(type='Bf',T0=T0,rate=0.2,T=330.0),
        dict(type='Bf',T0=T0,rate=0.05,T=345.0),
        dict(type='Bf',T0=T0,rate=0.02,T=275.0),
        dict(type='Bf',T0=T0,rate=0.01,T=310.0),
    ]

    dict(type='Bi',T0=T0,rate=0.2,T=330.0),
    dict(type='Bi',T0=T0,rate=0.05,T=140.0),
    dict(type='Bi',T0=T0,rate=0.01,T=275.0)

    cct = CCT(T0,Ac1,Ac3,transormation_data)
    cct.plot_CCT(colors,font_kwargs,ticks_kwargs,grid_kwargs,show=True)