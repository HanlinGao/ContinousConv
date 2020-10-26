import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle


def read_data(file):
    points = []
    with open(file, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                points.append(data[0].detach().numpy())
            except:
                break
    return points


points = read_data('predict.pkl')

for each in points:
    print('particle1 x ,y : ', each[0][0], each[0][1])
# fig = plt.figure()
# plt.xlim(-100, 100)
# plt.ylim(0, 500)
# point_ani, = plt.plot(points[0][:, 0], points[0][:, 1], "ro")
# # print(point_ani)
#
#
# def update_points(frame):
#     point_ani.set_data(points[frame][:, 0], points[frame][:, 1])
#     return point_ani,
#
#
# ani = FuncAnimation(fig, update_points, frames=len(points), blit=True)
# plt.show()