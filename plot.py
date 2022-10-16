import matplotlib.pyplot as plt
from typing import Tuple
import imageio
import numpy as np
import os


class Plot:


    def __init__(self, 
            figsize: Tuple[float, float],
            img_basename: str,
            x1: np.ndarray=None,
            x2: np.ndarray=None,
            f=None) -> None:
        self.img_paths = []
        self.img_basename = img_basename
        self.x1 = x1
        self.x2 = x2
        self.f = f
        plt.figure(figsize=figsize)


    def contour(self):
        assert self.x1 is not None, 'x1 array needed!'
        assert self.x2 is not None, 'x2 array needed!'
        assert self.f is not None, 'Test function needed!'
        x1_, x2_ = np.meshgrid(self.x1, self.x2)
        fx = x1_.copy()
        for i in range(x1_.shape[0]):
            for j in range(x1_.shape[1]):
                fx[i, j] = self.f(np.array([x1_[i, j], x2_[i, j]]))
        cp = plt.contourf(x1_, x2_, fx)
        plt.colorbar(cp)


    def point(self, x, color):
        plt.scatter(x[0], x[1], c=color)


    def save(self, img_name):
        img_path = self.img_basename + '-' + img_name + '.png'
        plt.savefig(img_path)
        self.img_paths.append(img_path)

    def gif(self):
        images = []
        for path in self.img_paths:
            images.append(imageio.imread(path))
            os.remove(path)
        imageio.mimsave(self.img_basename + '.gif', images, duration=0.5)


    def close(self):
        plt.close()





