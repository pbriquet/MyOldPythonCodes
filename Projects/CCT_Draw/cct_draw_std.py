
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import copy

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
    
    def __init__(self):
        self.T0 = 885.0
        self.Ac1 = 700.0
        self.Ac3 = 800.0
        self.cooling_rates = [100.0,50.0,20.0,10.0,5.0,1.0,0.5,0.1,0.05]
        self.cooling_rates = [100.0/2**k for k in range(0,12)]
    def plot_CCT(self,color_schema,font_schema,ticks_schema,grid_schema,**kwargs):
        colors = [(c['R']/255.0,c['G']/255.0,c['B']/255.0) for c in color_schema]
        lines_kwargs = {
            'Ac1':dict(color=colors[0]),
            'Ac3':dict(color=colors[0]),
            'Ms':dict(color='green'),
            'Mf':dict(color='darkgreen'),
            'Bs':dict(color='blueviolet'),
            'Bf':dict(color='indigo'),
            'B':dict(color='indigo'),
            'Ps':dict(color='teal'),
            'Pf':dict(color='teal')
        }
        
        schema=lines_kwargs
        image_formats = kwargs.get('image_formats',['png'])
        show = kwargs.get('show',True)
        dpi_choice = kwargs.get('dpi_choice',400)
        norm = kwargs.get('norm',False)
        
        lines_labes_fontschema = copy.copy(font_schema)
        lines_labes_fontschema['fontsize'] = 10

        fig = plt.figure(figsize=(12,8),dpi=100)
        ax = fig.add_subplot(111)
        time = np.logspace(-1,5,num=2000)
        
        cooling_curves = [self.T0 - i*time for i in self.cooling_rates]


        language = 'English'
        language_data = dict(
            English={'Xlabel':'t (s)','Ylabel':'T ' + r'$\mathrm{(°C)}$'},
        )
        plot_limits = {'X':[1e-1,1e4],'Y':[100.0,900.0]}


        ac3 = plt.hlines(self.Ac3,time[0],time[-1],schema['Ac3']['color'],label=r'$Ac_3$',zorder=10)
        ac1 = plt.hlines(self.Ac1,time[0],time[-1],schema['Ac1']['color'],label=r'$Ac_1$',zorder=10)
        ax.text(time[0], self.Ac1, ' ' + r'$Ac_1$' + ' = ' + str(self.Ac1) + ' °C', ha='left', va='bottom',color=schema['Ac1']['color'],**lines_labes_fontschema)
        ax.text(time[0], self.Ac3, ' ' + r'$Ac_3$' + ' = ' + str(self.Ac3) + ' °C', ha='left', va='bottom',color=schema['Ac3']['color'],**lines_labes_fontschema)

        lines = []
        for k in range(len(cooling_curves)):
            ax.plot(time,cooling_curves[k],'--',label=str(self.cooling_rates[k]) + ' °C/s',color='gray',alpha=0.2,lw=0.7)
            pos_x = (self.T0 - 200.0)/self.cooling_rates[k]
            ax.text(pos_x, 200.0 + 5.0 ,str(self.cooling_rates[k]) + ' °C/s', ha='left', va='baseline',fontsize=6,color='gray',zorder=10)

        ax.set_xscale('log')
        ax.grid(which='minor',**grid_schema['minor'])
        ax.grid(which='major',**grid_schema['major'])
        ax.set_axisbelow(True)
        #plt.figtext(0.53, 0.005, lower_text[language], wrap=True, horizontalalignment='center', fontproperties=font_schema['fontproperties'],fontsize=12)
        
        ax.set_xlabel(language_data[language]['Xlabel'],**font_schema)
        ax.set_ylabel(language_data[language]['Ylabel'],**font_schema)
        ax.set_xlim(plot_limits['X'][0],plot_limits['X'][1])
        ax.set_ylim(plot_limits['Y'][0],plot_limits['Y'][1])
        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) :    
            label.set_fontproperties(ticks_schema['fontproperties'])
            label.set_fontsize(ticks_schema['fontsize'])
        #plt.legend()
        
        plt.minorticks_on()
        plt.tight_layout()
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
        major=dict(b=True,lw=0.2,ls='-',color='gray'),
        minor=dict(b=False,lw=0.1,ls='-',color='gray',alpha=0.2),
        bar=dict(which='major',b=True,lw=0.2,ls=':',color='gray')
        )
    grid_kwargs = second_version_grid

    cct = CCT()
    cct.plot_CCT(colors,font_kwargs,ticks_kwargs,grid_kwargs,show=True)