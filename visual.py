from test_function import *
import numpy as np
import random 
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from itertools import zip_longest

def _cb(path=[]):
    def __cb(xk):
        path.append(np.copy(xk))
    return __cb


class AnimationContour(animation.FuncAnimation):
    def __init__(self, drawer, x0,  labels=[],  **kwargs):
        ax = drawer.get_test_func_plot()
        self.fig = ax.get_figure()
        self.ax = ax
        
        self.paths = [drawer.get_path_data(x0)]

        ff = max(path.shape[1] for path in self.paths)
  
        self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                      for _, label in zip_longest(self.paths, labels)]
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                       for line in self.lines]

        super(AnimationContour, self).__init__(self.fig, self.animate, init_func=self.init_anim,
                                                  frames=ff, interval=60, blit=True,
                                                  repeat_delay=5, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::,:i])
            point.set_data(*path[::,i-1:i])
        return self.lines + self.points


class Draw:
    def __init__(self, test_func = 'Himmelblau', opt_method = 'Newton-CG', type = 'contour', eps = 0.05):
        self.test_func = globals()[test_func]()
        self.f = globals()[test_func].f
        self.opt_method = opt_method
        self.type = type
        self.minx, self.miny, self.minz = self.test_func.get_minimum()
        self.eps = eps


    def get_x0(self):
        xran, yran = self.test_func.get_search_range()
        x0 = random.uniform(xran[0], xran[1])
        y0 = random.uniform(yran[0], yran[1])
        return np.array([x0, y0])

    def get_path_data(self, x0):
        _path = [x0]
        func = lambda args: self.f(*args)
        result = minimize(func, x0=x0, method=self.opt_method, tol=1e-20, callback=_cb(_path), options={'eps': self.eps})
        path = np.array(_path).T
        return path


    def get_test_func_plot(self):

        xran, yran = self.test_func.get_search_range()
        
        step = 0.1
        x, y = np.meshgrid(np.arange(xran[0], xran[1] + step, step), np.arange(yran[0], yran[1] + step, step))
        min_val = np.array([self.minx, self.miny]).reshape(-1, 1)
        z = self.test_func(x, y)
        if self.type == 'contour':
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
            ax.plot(*min_val, 'r*', markersize=10)
        elif self.type == '3d':
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection='3d', elev=50, azim=-50)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, edgecolor='none', alpha=0.8, norm = LogNorm(),cmap=plt.cm.jet)
            ax.plot(*min_val, self.test_func(*min_val), 'r*', markersize=10)
            ax.set_zlabel('$z$')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim(xran)
        ax.set_ylim(yran)
        return ax 


    def plot(self, x0):
        path = self.get_path_data(x0)
        
        xran, yran = self.test_func.get_search_range()
        
        step = 0.1
        x, y = np.meshgrid(np.arange(xran[0], xran[1] + step, step), np.arange(yran[0], yran[1] + step, step))

        z = self.test_func(x, y)

        min_val = np.array([self.minx, self.miny]).reshape(-1, 1)

        if self.type == 'contour':
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
            ax.quiver(path[0,:-1], path[1,:-1], path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
            ax.plot(*min_val, 'r*', markersize=10)
        elif self.type == '3d':
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection='3d', elev=50, azim=-50)
            ax.plot_surface(x, y, z, rstride=1, cstride=1, edgecolor='none', alpha=0.8, norm = LogNorm(),cmap=plt.cm.jet)
            ax.quiver(path[0,:-1], path[1,:-1], self.f(*path[::,:-1]), path[0,1:]-path[0,:-1], path[1,1:]-path[1,:-1], self.f(*(path[::,1:]-path[::,:-1])), color='k', normalize=True)
            ax.plot(*min_val, self.test_func(*min_val), 'r*', markersize=10)
            ax.set_zlabel('$z$')
        
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim(xran)
        ax.set_ylim(yran)

        plt.show()
