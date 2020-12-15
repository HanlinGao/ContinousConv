import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pickle
import os


def rendering_2d(box_file, fluids_file):
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)
        box = box_data[0][:, :2]

    points = []
    if os.path.splitext(fluids_file)[-1] == ".npy":

        with open(fluids_file, 'rb') as f:
            while True:
                try:
                    points.append(np.load(f))
                except:
                    break
    else:
        with open(fluids_file, 'rb') as f:
            data = pickle.load(f)
        print(len(data))
        for each in data:
            points.append(each[0])

    fig = plt.figure()
    point_ani, = plt.plot(points[0][:, 0], points[0][:, 1], "ro")
    plt.plot(box[:, 0], box[:, 1], "ko")

    def update_points(frame):
        point_ani.set_data(points[frame][:, 0], points[frame][:, 1])
        return point_ani,

    ani = animation.FuncAnimation(fig, update_points, frames=len(points), blit=True)
    plt.show()


def rendering_3d(box_file, fluids_file):
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    # np.savetxt('box_points.txt', box_data[0])
    box_x = []
    box_y = []
    box_z = []
    for each in box_data[0]:
        box_x.append(each[0])
        box_y.append(each[1])
        box_z.append(each[2])

    ax.scatter(box_x, box_z, box_y, s=50, c='gainsboro', alpha=0.4)
    # ax.scatter(lr_x, lr_z, lr_y, s=50, c='slateblue', alpha=0.4)
    # ax.scatter(bottom_x, bottom_z, bottom_y, s=50, c='darkcyan', alpha=0.4)
    # load fluid particles repeatedly since it was saved repeatedly with np.save
    fluids = []
    with open(fluids_file, 'rb') as f:
        while True:
            try:
                fluids.append(np.load(f))
            except:
                break

    # ax.set_xlim3d(-1, 1, 40)
    # ax.set_zlim3d(0, 4, 80)
    ax.set_ylim3d(-4, 4, 40)
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

    print(len(data))
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


# train on falling
rendering_2d('datasets/box_train.pkl', 'datasets/fall_eval.npy')
rendering_2d('datasets/box_train.pkl', 'datasets/fall_train.npy')
rendering_2d('datasets/box_train.pkl', 'datasets/falling.pkl')
# rendering_gt('datasets/box_train.pkl', 'datasets/falling.pkl')

# train on sliding
# rendering_3d('datasets/box_train.pkl', 'datasets/sliding_0.npy')
# rendering_gt('datasets/box_train.pkl', 'datasets/sliding.pkl')

# train on whole
# rendering_3d('datasets/box_train_4wall.pkl', 'datasets/whole_eval.npy')
# rendering_gt('datasets/box_train.pkl', 'datasets/3p_whole.pkl')

# rendering_3d('datasets/box_train.pkl', 'datasets/predict1.npy')
# rendering_3d('datasets/box_train.pkl', 'datasets/predict2.npy')
# rendering_3d('datasets/box_train.pkl', 'datasets/predict3.npy')
# rendering_gt('datasets/box_train.pkl', 'datasets/eval.pkl')