import os, sys
from time import sleep
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
list_files = []
for root, dirs, files in os.walk(__location__):
    for f in files:
        if(f.endswith('.tcm')):
            filepath = os.path.join(root,f)
            list_files.append(filepath)

for filepath in list_files:
    os.startfile(filepath,'open')
    sleep(10.0)
        
