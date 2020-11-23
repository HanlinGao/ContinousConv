
import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import pickle


def rendering_3d(box_file, fluids_file):
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)

    np.savetxt('box_points.txt', box_data[0])
    box_x = []
    box_y = []
    box_z = []
    for each in box_data[0]:
        if each[2] != -0.05:
            box_x.append(each[0])
            box_y.append(each[1])
            box_z.append(each[2])
    # box_x = box_data[0][:, 0]
    # box_y = box_data[0][:, 1]
    # box_z = box_data[0][:, 2]

    # load fluid particles repeatedly since it was saved repeatedly with np.save
    fluids = []
    with open(fluids_file, 'rb') as f:
        while True:
            try:
                fluids.append(np.load(f))
            except:
                break

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(box_x, box_z, box_y, s=50, c='gainsboro', alpha=0.4)
    ax.set_xlim3d(20, 80)
    ax.set_zlim3d(20, 80)
    ax.set_ylim3d(-10, 10)
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
        if each[2] != -0.05:
            box_x.append(each[0])
            box_y.append(each[1])
            box_z.append(each[2])

    # load fluid particles
    fluids = []
    with open(fluids_file, 'rb') as f:
        data = pickle.load(f)
    print('data shape', len(data), data)
    for each in data:
        fluids.append(each[0])

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(box_x, box_z, box_y, s=50, c='gainsboro', alpha=0.4)
    ax.set_xlim3d(20, 80)
    ax.set_zlim3d(20, 80)
    ax.set_ylim3d(-10, 10)
    print(len(fluids))
    def update_graph(frame):
        data = fluids[frame]
        graph.set_data(data[:, 0], data[:, 2])
        graph.set_3d_properties(data[:, 1])
        return graph,

    data = np.array(fluids[0])
    print(data[0][2])
    graph, = ax.plot(data[:, 0], data[:, 2], data[:, 1], color='coral', marker='o', fillstyle='full',
                     markeredgecolor='r', markeredgewidth=0.2, linestyle="")
    ani = animation.FuncAnimation(fig, update_graph, len(fluids), blit=True)

    plt.show()

# rendering_3d('dataset/input_train/box_train.pkl', 'dataset/input_train/119_trian.npy')
# rendering_3d('../dataset/input_train/box_train.pkl', '../dataset/out/predicts.npy')

# rendering_3d('../dataset/input_train/box_train.pkl', '../dataset/out/predicts_10p_200ts.npy')
rendering_gt('../dataset/input_train/box_train.pkl', '../dataset/input_train/train_10p_long2.pkl')
# rendering_3d('datasets/box_train.pkl', 'datasets/117_199_trian.npy')
# rendering_gt('datasets/box_train.pkl', 'datasets/fluid_train.pkl')

