# xinyao: this file contains test functions for optimization algo
# notice that we only visulize in 3-dimensional space 
import numpy as np

class Rastrigin(object):
    f = lambda x, y:10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
    def __call__(self, x, y):
        return 10 * 2 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
    
    def compute(self, x, y):
        return self.__call__(x, y)
        
    @staticmethod
    def get_minimum():
        return 0, 0, 0  # x y z
    @staticmethod
    def get_search_range():
        return [-5.12, 5.12], [-5.12, 5.12]


class Ackley(object):
    f = lambda x, y: -20 * np.exp(-0.2 * (0.5*(x**2 + y**2))**0.5) - np.exp(0.5* (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e
    def __call__(self, x, y):
        return -20 * np.exp(-0.2 * (0.5*(x**2 + y**2))**0.5) - np.exp(0.5* (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + 20 + np.e
   
    def compute(self, x, y):
        return self.__call__(x, y)
        
    @staticmethod
    def get_minimum():
        return 0, 0, 0
    @staticmethod
    def get_search_range():
        return [-5, 5], [-5, 5]


class Sphere(object):
    f = lambda x, y: x **2 + y ** 2
    def __call__(self, x, y):
        return x **2 + y ** 2

    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 0, 0, 0
    @staticmethod
    def get_search_range():
        return [-5, 5], [-5, 5]
    

class Rosenbrock(object):
    f = lambda x, y: 100 * (y - x**2) ** 2 + (1 - x) ** 2
    def __call__(self, x, y):
        return 100 * (y - x**2) ** 2 + (1 - x) ** 2
   
    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 0,0,0
    @staticmethod
    def get_search_range():
        return [-5, 5], [-5, 5]


class Beale(object):
    f = lambda x, y: (1.5-x+x*y) ** 2 + (2.25 - x + x*y**2) ** 2  + (2.625 - x + x*y**3) ** 2
    def __call__(self, x, y):
        return (1.5-x+x*y) ** 2 + (2.25 - x + x*y**2) ** 2  + (2.625 - x + x*y**3) ** 2
  
    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 3, 0.5, 0
    @staticmethod
    def get_search_range():
        return [-4.5, 4.5], [-4.5, 4.5]


class Booth(object):
    f = lambda x, y: (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    def __call__(self, x, y):
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 1, 3, 0
    @staticmethod
    def get_search_range():
        return [-10, 10], [-10, 10]


class GoldsteinPrice(object):
    f = lambda x, y: (1 + (x + y + 1)**2 * (19-14 * x + 3 * x**2 - 14 * y + 6*x*y + 3*y**2)) * (30 + (2*x -3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    def __call__(self, x, y):
        return (1 + (x + y + 1)**2 * (19-14 * x + 3 * x**2 - 14 * y + 6*x*y + 3*y**2)) * (30 + (2*x -3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 0, -1, 3
    @staticmethod
    def get_search_range():
        return [-2, 2], [-2, 2]


class Matyas(object):
    f = lambda x, y: 0.26*(x**2 + y**2) - 0.48* x * y
    def __call__(self, x, y): 
        return 0.26*(x**2 + y**2) - 0.48* x * y

    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return 0, 0, 0
    @staticmethod
    def get_search_range():
        return [-10, 10], [-10, 10]


class Himmelblau(object):
    f  = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7) **2

    def __call__(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7) **2

    def compute(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7) **2

    @staticmethod
    def get_minimum():
        return 3, 2, 0
    @staticmethod
    def get_search_range():
        return [-5, 5], [-5, 5]


class McCormick(object):
    f = lambda x, y: np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1
    def __call__(self, x, y):
        return np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1

    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return -0.54719, -1.54719, -1.9133
    @staticmethod
    def get_search_range():
        return [-1.5, 4], [-3, 4]


class StyblinskiTang(object):
    f = lambda x, y: np.sin(x + y) + (x - y)**2 - 1.5*x + 2.5*y + 1
    def __call__(self, x, y):
        return (x**4 - 16 * x**2 + 5 * x)/2 + (y**4 - 16*y**2 + 5 * y)
    
    def compute(self, x, y):
        return self.__call__(x, y)

    @staticmethod
    def get_minimum():
        return -2.903534, -2.903534, -39.16616 * 2
    @staticmethod
    def get_search_range():
        return [-5, 5], [-5, 5]
