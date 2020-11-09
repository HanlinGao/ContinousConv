import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle


def read_predict_data(file):
    points = []
    with open(file, 'rb') as f:
        while True:
            try:
                data = pickle.load(f)
                points.append(data[0].detach().numpy())
            except:
                break
    return points


def read_origin_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        box = np.array(data[0][3])    # box remains the same, so just choose the 0th timestep
        # box_normals = data[0][4]    # box_normals remains the same
        points = [np.array(x) for x in data[:, 0]]
        # points = np.array(data[:, 0])     # pos for each timestep
    return box, points


def read_numpy_data(file):
    points = []
    with open(file, 'rb') as f:
        while True:
            try:
                data = np.load(f)
                points.append(data)
            except:
                break
    return points


# box, points = read_origin_data('origin.pkl')
with open('box_train.pkl', 'rb') as f:
    box_data = pickle.load(f)
    box = box_data[0]
points = read_numpy_data('prediction.npy')
print('box', box)
print('points', points[0])

fig = plt.figure()
# plt.xlim(-100, 100)
# plt.ylim(0, 500)
point_ani, = plt.plot(points[0][:, 0], points[0][:, 1], "ro")
plt.plot(box[:, 0], box[:, 1], "go")


def update_points(frame):
    point_ani.set_data(points[frame][:, 0], points[frame][:, 1])
    return point_ani,

ani = FuncAnimation(fig, update_points, frames=len(points), blit=True)
plt.show()