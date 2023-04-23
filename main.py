import numpy as np
from matplotlib import pyplot as plt
from utils import *
map_xLim = [-150, 100]
map_yLim = [-100, 100]
robot_init = [0, 0, 0]

way_points = np.array([[10, -40], [40, -60], [80, -50], [90, -20],
                       [60, 0], [70, 30], [70, 70], [30, 80],
                       [10, 50], [-40, 70], [-80, 72], [-90, 40],
                       [-110, 5], [-80, -30], [-80, -70], [-40, -80],
                       [-20, -50]])
min_distance_way_points = 5




if __name__ == '__main__':
    x = np.array([0,0,0])
    C_sqrt = np.array([[1,2],[0,3]])
    C = C_sqrt.dot(C_sqrt.T)
    s = 5.991
    sqrt_s = np.sqrt(s)

    sample = C_sqrt.dot(np.random.randn(2, 100))+x[:2,None]
    sample2 = np.random.multivariate_normal(x[:2],C,100).T
    plt.figure()
    plt.scatter(sample[0, :],sample[1, :])
    plt.scatter(sample2[0, :],sample2[1, :], c="red")
    # drawEllipse(x, 1*sqrt_s,3*sqrt_s, "b")
    drawProbEllipse(x,C,s,"yellow")
    # old1_drawProbEllipse(x,C,s,"blue")
    # old_drawProbEllipse(x,C,s,"green")
    # drawEllipse(x, 9,9, "green")
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.show()

    # plt.figure()
    # plt.plot(way_points[:, 0], way_points[:, 1])
    # plt.xlim(map_xLim)
    # plt.ylim(map_yLim)
    # plt.xlabel("meters")
    # plt.ylabel("meters")
    # plt.show()
