import numpy as np
from matplotlib import pyplot as plt

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
    plt.figure()
    plt.plot(way_points[:, 0], way_points[:, 1])
    plt.xlim(map_xLim)
    plt.ylim(map_yLim)
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.show()
