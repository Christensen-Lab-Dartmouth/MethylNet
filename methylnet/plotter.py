"""
plotter.py
=======================
Plotting mechanisms for training that are now defuct.
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd, numpy as np

class PlotTransformer:
    """Plotting Transformer to help with plotting embeddings changing over epochs; defunct."""
    def __init__(self, data, color_list):
        self.data=data
        self.color_list=color_list
    def transform(self):
        return pd.DataFrame(np.vstack((PCA(n_components=2).fit_transform(self.data).T,self.color_list)).T,columns=['x','y','color'])

class Plot:
    """Stores plotting information; defunct, superceded, see methylnet-visualize."""
    def __init__(self, title, xlab='vae1', ylab='vae2', data=[]):
        self.title=title
        self.xlab=xlab
        self.ylab=ylab
        self.data=data

    def return_img(self,ax,animated=False):

        scatter = ax.scatter(x=self.data['x'],y=self.data['y'],s=5,c=self.data['color'] if 'color' in list(self.data) else None)
        plt.title(self.title)
        plt.xlabel(self.xlab)
        plt.ylabel(self.ylab)
        plt.xlim(self.data['x'].min(),self.data['x'].max())
        plt.ylim(self.data['y'].min(),self.data['y'].max())
        #if animation:
        #    return [plt.imshow(animated=True)]
        return scatter

class Plotter:
    """Plot embeddings and training curve from Plot objects; defunct, superceded, see methylnet-visualize."""
    def __init__(self,plots,animation=True):
        self.plots=plots
        self.animation=animation

    def animate(self, i):
        self.ax.clear()
        plt.title(self.plots[i].title)
        plt.xlabel(self.plots[i].xlab)
        plt.ylabel(self.plots[i].ylab)
        self.scatter = self.ax.scatter(x=self.plots[i].data['x'],y=self.plots[i].data['y'],s=5,c=self.plots[i].data['color'] if 'color' in list(self.plots[i].data) else None)#.set_data(self.plots[i].data['x'],self.plots[i].data['y'])#=self.plots[i].return_img(self.ax,True)#.set_data(self.plots[i].data['x'],self.plots[i].data['y'])
        return self.scatter,

    def write_plots(self, output_fname):
        if self.animation:
            animate = lambda i: self.animate(i)
            fig=plt.figure()
            self.ax=plt.axes()
            self.scatter = self.ax.scatter([],[])
            def init():
                self.scatter.set_data([],[])
                return self.scatter,
            ani=animation.FuncAnimation(fig,animate,frames=len(self.plots),interval=100,blit=True)#[plot.return_img(ax,True) for plot in self.plots]
            ani.save(output_fname,writer='ffmpeg')
        else:
            f, axes = plt.subplots(len(self.plots),1,sharex=True,sharey=False)
            for i in range(len(self.plots)):
                self.plots[i].return_img(axes[i])
            plt.savefig(output_fname,dpi=300,figsize=(7,7))
