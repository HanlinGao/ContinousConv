import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pickle


# def read_predict_data(file):
#     points = []
#     with open(file, 'rb') as f:
#         while True:
#             try:
#                 data = pickle.load(f)
#                 points.append(data[0].detach().numpy())
#             except:
#                 break
#     return points


# def read_origin_data(file):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         box = np.array(data[0][3])    # box remains the same, so just choose the 0th timestep
#         # box_normals = data[0][4]    # box_normals remains the same
#         points = [np.array(x) for x in data[:, 0]]
#         # points = np.array(data[:, 0])     # pos for each timestep
#     return box, points


# def read_numpy_data(file):
#     points = []
#     with open(file, 'rb') as f:
#         while True:
#             try:
#                 data = np.load(f)
#                 points.append(data)
#             except:
#                 break
#     return points


# # box, points = read_origin_data('origin.pkl')
# with open('box_train.pkl', 'rb') as f:
#     box_data = pickle.load(f)
#     box = box_data[0]
# points = read_numpy_data('prediction.npy')
# print('box', box)
# print('points', points[0])
#
# fig = plt.figure()
# # plt.xlim(-100, 100)
# # plt.ylim(0, 500)
# point_ani, = plt.plot(points[0][:, 0], points[0][:, 1], "ro")
# plt.plot(box[:, 0], box[:, 1], "go")
#
#
# def update_points(frame):
#     point_ani.set_data(points[frame][:, 0], points[frame][:, 1])
#     return point_ani,
#
# ani = FuncAnimation(fig, update_points, frames=len(points), blit=True)
# plt.show()

def rendering_3d(box_file, fluids_file):
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # np.savetxt('box_points.txt', box_data[0])
    back_box_x = []
    back_box_y = []
    back_box_z = []

    lr_x = []
    lr_y = []
    lr_z = []

    bottom_x = []
    bottom_y = []
    bottom_z = []
    for each in box_data[0]:
        if each[2] == 0.1:
            back_box_x.append(each[0])
            back_box_y.append(each[1])
            back_box_z.append(each[2])

        if each[0] == -1 or each[0] == 1:
            lr_x.append(each[0])
            lr_y.append(each[1])
            lr_z.append(each[2])

        if each[1] == 0:
            bottom_x.append(each[0])
            bottom_y.append(each[1])
            bottom_z.append(each[2])

    ax.scatter(back_box_x, back_box_z, back_box_y, s=50, c='gainsboro', alpha=0.4)
    ax.scatter(lr_x, lr_z, lr_y, s=50, c='slateblue', alpha=0.4)
    ax.scatter(bottom_x, bottom_z, bottom_y, s=50, c='darkcyan', alpha=0.4)
    # load fluid particles repeatedly since it was saved repeatedly with np.save
    fluids = []
    with open(fluids_file, 'rb') as f:
        while True:
            try:
                fluids.append(np.load(f))
            except:
                break

    ax.set_xlim3d(-1, 1, 40)
    ax.set_zlim3d(0, 4, 80)
    ax.set_ylim3d(-1, 1, 40)
    print(len(fluids))
    def update_graph(frame):
        data = fluids[frame]
        graph.set_data(data[:, 0], data[:, 2])
        graph.set_3d_properties(data[:, 1])
        return graph,

    data = np.array(fluids[0])
    graph, = ax.plot(data[:, 0], data[:, 2], data[:, 1], color='coral', marker='o', fillstyle='full',
                     markeredgecolor='r', markeredgewidth=0.2, linestyle="")
    ani = animation.FuncAnimation(fig, update_graph, len(fluids), blit=True)

    plt.show()


def rendering_gt(box_file, fluids_file):
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)

    # np.savetxt('box_points.txt', box_data[0])
    box_x = []
    box_y = []
    box_z = []
    for each in box_data[0]:
        box_x.append(each[0])
        box_y.append(each[1])
        box_z.append(each[2])

    # load fluid particles
    fluids = []
    with open(fluids_file, 'rb') as f:
        data = pickle.load(f)

    for each in data:
        fluids.append(each[0])

    print(fluids)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(box_x, box_z, box_y, s=50, c='gainsboro', alpha=0.4)
    # ax.set_xlim3d(-1, 1, 40)
    # ax.set_zlim3d(0, 4, 80)
    # ax.set_ylim3d(-1, 1, 40)
    print(len(fluids))
    def update_graph(frame):
        data = fluids[frame]
        graph.set_data(data[:, 0], data[:, 2])
        graph.set_3d_properties(data[:, 1])
        return graph,

    data = np.array(fluids[0])
    graph, = ax.plot(data[:, 0], data[:, 2], data[:, 1], color='coral', marker='o', fillstyle='full',
                     markeredgecolor='r', markeredgewidth=0.2, linestyle="")
    ani = animation.FuncAnimation(fig, update_graph, len(fluids), blit=True)

    plt.show()

# rendering_3d('datasets/box_train.pkl', 'datasets/119_trian.npy')
# rendering_3d('datasets/box_train.pkl', 'datasets/renderdata.npy')
# rendering_3d('datasets/box_train.pkl', 'datasets/117_199_trian.npy')
# rendering_gt('datasets/box_train.pkl', 'datasets/fluid_train_20_80.pkl')
rendering_gt('datasets/box_train_box.pkl', 'datasets/fluid_train_3p_-5_-5.pkl')