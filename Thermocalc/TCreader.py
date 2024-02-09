from math import *
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys, os
import pandas as pd
import copy
import numpy as np
import random
from collections import OrderedDict
from labellines import labelLine, labelLines

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
scientific_match = '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'
float_match = '[-+]?[0-9]*\.?[0-9]+'
numbers_match = scientific_match + '|' + float_match
# [-+] faz aceitar tanto positivo quanto negativo no inicio, porem, "?" coloca como opcional nos casos que nao tenham.
# O [\d] faz aceitar um digito no inicio da notacao cientifica. + adiciona o . (\.) no seguinte, porem deixa opcinal esta parte caso o numero seja inteiro.
# Em seguida o [\d]* permite inumeros numeros nas casas decimais posteriores.
# [Ee] permite que a notacao cientifica esteja com e ou E

class ScheilReader:
    # GOC Code
    
    def __init__(self,exp_filepath,verbose=True):
        self.verbose = verbose  # verbose mode
        self.exp_filepath = exp_filepath    # path of exp file
        self.xscale, self.yscale = 0, 0     
        self.xlength, self.ylength = 11.5,11.5
        self.xtype, self.ytype = 'lin', 'log'
        self.xtext, self.ytext = 'T (C)', 'BPW (*)'
        self.xlength, self.ylength = 11.5, 11.5
        self.index_names = {}
        self.names_index = {}
        self.read()

    def read(self):
        file0 = open(self.exp_filepath,'r')
        phase = 0
        clip_on = False
        block_region = False
        header = True
        # Search for data columns
        clipoff = False
        clip = False
        block = False
        phase = None
        self.index_names = {}
        self.names_index = {}
        for line in file0:
            _line = line.rstrip()
            
            if(clipoff): # Na linha posterior ao clip off, esta a informacao relevante
                search = re.search(r'(\d+):[ ]?([\s\S]+)',_line)
                number = search.group(1)
                names = search.group(2).split(',')
                self.index_names[str(number)] = names[0] if len(names) == 1 else names[1]
                self.names_index[names[0]] = str(number)
                clipoff = False
            if(re.match(r'\s*BLOCK',_line)):
                block = True
            elif(re.match(r'\s*BLOCKEND',_line)):
                block = False
            if(block):
                if(re.match(r'\s*CLIP OFF',_line)):
                    clipoff = True
        print(self.index_names)
        input()
        self.data = {}
        for k,v in self.index_names.items():
            self.data[k] = {'name':v,'X':[],'Y':[]}
        file0.close()
        file0 = open(self.exp_filepath,'r')
        for line in file0:
            _line = line.rstrip()
            if(header):
                if(re.match(r'\s*PROLOG',line)):
                    self.prolog = [int(x) for x in re.findall(numbers_match,line)][0]  # Leitura apos prolog int
                    if(self.verbose):
                        print('prolog = ' + str(self.prolog))
                elif(re.match(r'\s*XSCALE',line)):
                    self.xscale = [float(x) for x in re.findall(numbers_match,line)]
                    if(self.verbose):
                        print('xscale = ' + str(self.xscale))
                elif(re.match(r'\s*YSCALE',line)):
                    self.yscale = [float(x) for x in re.findall(numbers_match,line)]
                    if(self.verbose):
                        print('yscale = ' + str(self.yscale))
                elif(re.match(r'\s*XTYPE',line)):
                    self.xtype = re.search(r'\s*XTYPE\s*(\w+)',_line).group(1)
                    if(self.verbose):
                        print('xtype = ' + str(self.xtype))
                elif(re.match(r'\s*YTYPE',line)):
                    self.ytype = re.search(r'\s*YTYPE\s*(\w+)',_line).group(1)
                    if(self.verbose):
                        print('ytype = ' + str(self.ytype))
                elif(re.match(r'\s*XLENGTH',line)):
                    self.xlength = [float(x) for x in re.findall(numbers_match,line)][0]
                    if(self.verbose):
                        print('xlength = ' + str(self.xlength))
                elif(re.match(r'\s*YLENGTH',line)):
                    self.ylength = [float(x) for x in re.findall(numbers_match,line)][0]
                    if(self.verbose):
                        print('ylength = ' + str(self.ylength))
                elif(re.match(r'\s*XTEXT',line)):
                    self.xtext = re.findall('\w+',line)[1]
                    if(self.verbose):
                        print('xtext = ' + str(self.xtext))
                elif(re.match(r'\s*YTEXT',line)):
                    self.ytext = re.findall('\w+',line)[1]
                    header = False
                    if(self.verbose):
                        print('ytext = ' + str(self.ytext))
            else:
                if(self.verbose):
                    print(_line)
                if(re.match(r'^\s*CLIP OFF',line)):
                    clip = False
                if(phase):
                    if(clip and block_region):
                        data = [float(x) for x in re.findall(numbers_match,line)]
                        if(len(data)>1):
                            self.data[phase]['X'].append(data[0])
                            self.data[phase]['Y'].append(data[1])
                if(re.match(r'^\s*BLOCK ',line)):
                    block_region = True
                elif(re.match(r'^\s*BLOCKEND',line)):
                    block_region = False
                    phase = None

                if(block_region):
                    if(re.search(r"MWA'",_line)):
                        phase = re.search(r"MWA\'(\d+)",_line).group(1)
                if(re.match(r'\s*CLIP',line)):
                        if(re.match(r'^\s*CLIP ON',line)):
                            clip = True
                        elif(re.match(r'^\s*CLIP OFF',line)):
                            clip = False
        self.names_plot = {}
        for k,v in self.index_names.items():
            self.names_plot[v] = v

    def plot(self):
        cols = []
        tmp_dictionary = {}
        max_row = 0
        for k,v in self.data.items():
            if(max_row < len(v['X'])):
                max_row = len(v['X'])
        for k,v in self.data.items():
            v['X'].extend([np.nan]*(max_row - len(v['X'])))
            v['Y'].extend([np.nan]*(max_row - len(v['Y'])))
            tmp_dictionary[v['name'] + ':X'] = v['X']
            tmp_dictionary[v['name'] + ':Y'] = v['Y']
        for k,v in self.data.items():
            tmp_d = []
            for row in range(len(v['X'])):
                if(row!=0 and v['X'][row] != np.nan):
                    tmp_d.append( np.sqrt((v['X'][row] - v['X'][row - 1] )**2 + (v['Y'][row] - v['Y'][row - 1] )**2))
                else:
                    tmp_d.append(0.0)
            tmp_dictionary[v['name'] + ':Z'] = tmp_d
        df = pd.DataFrame.from_dict(tmp_dictionary)
        
        stop_indexes = {}
        lines = {}
        # lines has keys of phases, and an array of dictionaries with segments to be plotted.
        for k,v in self.data.items():
            df_new = df.loc[df[v['name'] + ':Z'] > df[v['name'] + ':Z'].mean() + 1.0*df[v['name'] + ':Z'].std()]    # mean + std is alright
            stop_indexes[v['name']] = list(df_new.index.get_values())
            stop_indexes[v['name']].append(max_row)
            lines[v['name']] = []
            last_stop = 0

            for i,index in enumerate(stop_indexes[v['name']]):
                lines[v['name']].append({'X':df.iloc[last_stop:index][v['name'] + ':X'],'Y':df.iloc[last_stop:index][v['name'] + ':Y']})
                last_stop = index
        

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Make a color cycle with a colormap (tab20)
        cycle = list(cm.tab20(np.linspace(0,0.9,len(lines.keys()))))
        random.seed(5)
        random.shuffle(cycle)
        i_cycle = 0
        for j,phase in lines.items():
            for i,k in enumerate(phase):
                ax.plot(k['X'],k['Y'],color=cycle[i_cycle],label=self.names_plot[j])
            i_cycle +=1
        
        #ax.set_xlabel(r'$\% Nb$' + ' ' + r'$(\% wt)$')
        #ax.set_ylabel(r'$T$' + ' ' + r'$(°C)$')

        # Set legend position
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))    # Put legends of lines with the same label
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.75)) 
        
        ax.set_xlim(400.0,1600.0)
        ax.set_ylim(1e-4,1.0)
        ax.set_yscale('log')
        ax.grid(b=True,which='major',linestyle='-',color='darkgrey')
        ax.grid(b=True,which='minor',linestyle='--',color='lightgrey')
        plt.minorticks_on()
        plt.show()

            

if __name__=='__main__':
    folderpath = os.path.join(__location__,'Bonon\\Eq')
    file0 = os.path.join(folderpath,'bpw.exp')
    e = ScheilReader(file0,verbose=True)   
    e.plot()