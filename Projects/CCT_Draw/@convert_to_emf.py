import matplotlib.pyplot as plt
import subprocess
import os
inkscapePath = r"E:\@ Apps\InkscapePortable\InkscapePortable.exe"

def exportEmf(folderpath, plotName, fig=None, keepSVG=False):
    """Save a figure as an emf file

    Parameters
    ----------
    savePath : str, the path to the directory you want the image saved in
    plotName : str, the name of the image 
    fig : matplotlib figure, (optional, default uses gca)
    keepSVG : bool, whether to keep the interim svg file
    """

    print(plotName)
    figFolder = folderpath + r"\{}.{}"
    svgFile = os.path.join(folderpath,plotName)
    emfFile = os.path.join(folderpath,plotName + '.emf')


    if fig:
        use=fig
    else:
        use=plt
    #use.savefig(svgFile)
    subprocess.run([inkscapePath, svgFile, '-M', emfFile])

    if not keepSVG:
        os.system('del "{}"'.format(svgFile))

if __name__ == "__main__":
    __loc__ = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
    for root,dirs,files in os.walk(__loc__):
        for f in files:
            if(f.endswith('svg')):
                exportEmf(__loc__,f,keepSVG=True)