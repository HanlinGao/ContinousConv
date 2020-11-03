import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_particle_data(file, save_file, num_lines=10):
    # map origin data -> [[pos_x, pos_y, pos_z], [vel_x, vel_y, vel_z], [x_label, y_label, z_label]]
    # suitable for training data, but not for reading by human
    dataset = []
    with open(file, "r") as f:
        # get one time step data
        for each_step in range(num_lines+2):
            line = f.readline()

            pos = []    # in the form of [[pos_1, ], [pos_2, ], ..[pos_n, ]]
            vel = []    # in the form of [[vel_1, ], [vel_2, ], ..[vel_n, ]]
            particles = line.rstrip().split('|')

            # since some lines end with '|', which will generate '', needing filtering
            particles = list(filter(lambda x: x, particles))

            for each_particle in particles:
                p_data = each_particle.strip().split('\t')
                p_pos = [float(p_data[0]), float(p_data[1]), 0.]
                p_vel = [float(p_data[2]), float(p_data[3]), 0.]

                pos.append(p_pos)
                vel.append(p_vel)

            step = [np.array(pos), np.array(vel)]
            dataset.append(step)

    # print(dataset[0][0])
    # add labels, so for each time step in dataset, the form will be [pos_matrix, vel_matrix, label_matrix(t+1), label_matrix(t+2)]
    for j in range(num_lines):
        dataset[j].append(dataset[j+1][0][:])
        dataset[j].append(dataset[j+2][0][:])
        # print(len(dataset))
        # print(dataset[0])
    # save into a .pkl file
    with open(save_file, 'wb') as f:
        pickle.dump(dataset[: -2], f)
    return


def apic_2d_data_clean(origin_file, save_file):
    # map origin data -> 2D array [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, label_x, label_y, label_z]
    # easy for reading by human with numpy savetxt method
    with open(origin_file, 'rb') as f:
        dataset = pickle.load(f)

    # splice the dataset into a 2D array form
    res = []
    for each in dataset:    # dataset: [[pos_matrix, vel_matrix, label_matrix], [s..t..e..p..2], ..., ]
        res.append(np.hstack((each[0], each[1], each[2])))

    res = np.concatenate(res)
    np.savetxt(save_file, res)

    return


def data_normalization(origin_file, save_file):
    # as for initialization, make x, z in range of [-0.5, 0.5], y in range of [0, 1.5]
    # core idea and goal is to make the whole range as pos in [-1, 1], vel in [-4, 4]
    x_target_l = -1
    x_target_up = 1

    y_target_l = 0
    y_target_up = 4

    vel_target_l = -4
    vel_target_up = 4

    with open(origin_file, 'rb') as f:
        dataset = pickle.load(f)

    # calculate situation of pos
    x_min, x_max, x_sum = 100, -100, 0
    y_min, y_max, y_sum = 100, -100, 0
    z_min, z_max, z_sum = 100, -100, 0
    count = 0
    for each_step in dataset:   # [pos_matrix, vel_matrix, label_matrix(t+1), label_matrix(t+2)]
        for particle in each_step[0]:   # [x, y, z]
            count += 1
            if particle[0] < x_min:
                x_min = particle[0]
            if particle[0] > x_max:
                x_max = particle[0]
            x_sum += particle[0]

            if particle[1] < y_min:
                y_min = particle[1]
            if particle[1] > y_max:
                y_max = particle[1]
            y_sum += particle[1]

            if particle[2] < z_min:
                z_min = particle[2]
            if particle[2] > z_max:
                z_max = particle[2]
            z_sum += particle[2]

    print('positions situation: ')
    print('x_max: ', x_max, 'x_min: ', x_min, 'x_mean: ', x_sum/count)
    print('y_max: ', y_max, 'y_min: ', y_min, 'y_mean: ', y_sum/count)
    print('z_max: ', z_max, 'z_min: ', z_min, 'z_mean: ', z_sum/count)
    # calculate situation of velocity
    vel_min = 100
    vel_max = -100
    for each_step in dataset:
        max_temp = np.max(each_step[1])
        min_temp = np.min(each_step[1])
        if max_temp > vel_max:
            vel_max = max_temp
        if min_temp < vel_min:
            vel_min = min_temp

    print(dataset[2][0])
    res = []
    # normalize x into [-1, 1], y into [0, 4]
    for each_step in dataset:   # [pos_matrix, vel_matrix, label_matrix(t+1), label_matrix(t+2)]
        # normalize pos_matrix
        each_step[0][:, 0] = x_target_l + (x_target_up - x_target_l) / (x_max - x_min) * (each_step[0][:, 0] - x_min)
        each_step[0][:, 1] = y_target_l + (y_target_up - y_target_l) / (y_max - y_min) * (each_step[0][:, 1] - y_min)

        # normalize label_matrix
        each_step[2][:, 0] = x_target_l + (x_target_up - x_target_l) / (x_max - x_min) * (each_step[2][:, 0] - x_min)
        each_step[2][:, 1] = y_target_l + (y_target_up - y_target_l) / (y_max - y_min) * (each_step[2][:, 1] - y_min)

        each_step[3][:, 0] = x_target_l + (x_target_up - x_target_l) / (x_max - x_min) * (each_step[3][:, 0] - x_min)
        each_step[3][:, 1] = y_target_l + (y_target_up - y_target_l) / (y_max - y_min) * (each_step[3][:, 1] - y_min)

        # normalize vel_matrix
        each_step[1][:, :2] = vel_target_l + (vel_target_up - vel_target_l) / (vel_max - vel_min) * \
                              (each_step[1][:, :2] - vel_min)

        res.append(each_step)
    # save the normalized trianing data into .pkl file
    print(res[2][0])
    with open(save_file, 'wb') as f:
        pickle.dump(res, f)

    return


def boundary_generation(box_save_file):
    # generate a list in the form of [box_pos_matrix, box_normals_matrix]
    box_pos = []
    box_normals = []

    # generate the bottom
    bottom_x = np.arange(-1, 1, 0.05)
    for each in bottom_x:
        box_pos.append([each, 0., 0.])
        box_normals.append([0., 1., 0.])

    # generate front and back
    x = np.arange(-1, 1, 0.05)
    y = np.arange(0, 4, 0.05)
    xv, yv = np.meshgrid(x, y)

    for each_pair in zip(xv.flatten(), yv.flatten()):
        box_pos.append([each_pair[0], each_pair[1], 0.05])
        box_normals.append([0., 0., -1.])

        box_pos.append([each_pair[0], each_pair[1], -0.05])
        box_normals.append([0., 0., 1.])

    # generate left and right
    for each in y:
        box_pos.append([1., each, 0.])
        box_normals.append([-1., 0., 0.])

        box_pos.append([-1., each, 0.])
        box_normals.append([1., 0., 0.])

    box_pos = np.array(box_pos)
    box_normals = np.array(box_normals)
    # np.savetxt('box_pos_txt.txt', box_pos)
    # np.savetxt('box_noramls_txt.txt', box_normals)

    with open('box_train.pkl', 'wb') as f:
        pickle.dump([box_pos, box_normals], f)


def boundary_generation_no_wall(box_save_file):
    # generate a list in the form of [box_pos_matrix, box_normals_matrix]
    box_pos = []
    box_normals = []

    # generate the bottom
    bottom_x = np.arange(-1, 1, 0.05)
    for each in bottom_x:
        box_pos.append([each, 0., 0.])
        box_normals.append([0., 1., 0.])

    y = np.arange(0, 4, 0.05)
    # generate left and right
    for each in y:
        box_pos.append([1., each, 0.])
        box_normals.append([-1., 0., 0.])

        box_pos.append([-1., each, 0.])
        box_normals.append([1., 0., 0.])

    box_pos = np.array(box_pos)
    box_normals = np.array(box_normals)
    # np.savetxt('box_pos_txt.txt', box_pos)
    # np.savetxt('box_noramls_txt.txt', box_normals)

    with open('box_train_no_wall.pkl', 'wb') as f:
        pickle.dump([box_pos, box_normals], f)


def normal_analysis(normals_file, box_pos_file):
    # render the data used by pretrained_model, analyse the setting of normals
    with open(normals_file, 'r') as f_normals:
        with open(box_pos_file, 'r') as f_pos:
            set_x_1 = []
            set_x_n_1 = []
            set_y_1 = []
            set_y_n_1 = []
            set_z_1 = []
            set_z_n_1 = []
            others = []
            for each_line in f_normals:
                data = each_line.split(',')
                pos = f_pos.readline().split(',')

                data = list(map(lambda x: float(x), data))
                pos = list(map(lambda x: float(x), pos))

                if data[0] == 1:
                    set_x_1.append([pos[0], pos[1], pos[2]])
                elif data[0] == -1:
                    set_x_n_1.append([pos[0], pos[1], pos[2]])
                elif data[1] == 1:
                    set_y_1.append([pos[0], pos[1], pos[2]])
                elif data[1] == -1:
                    set_y_n_1.append([pos[0], pos[1], pos[2]])
                elif data[2] == 1:
                    set_z_1.append([pos[0], pos[1], pos[2]])
                elif data[2] == -1:
                    set_z_n_1.append([pos[0], pos[1], pos[2]])
                else:
                    others.append([pos[0], pos[1], pos[2]])

    # visualize them in 3D space to see the pattern
    # box particles whose 1st dimension is set to be 1
    set_x_1 = np.array(set_x_1)
    x_1 = set_x_1[:, 0]
    y_1 = set_x_1[:, 1]
    z_1 = set_x_1[:, 2]
    # box particles whose 1st dimension is set to be -1
    set_x_n_1 = np.array(set_x_n_1)
    x_2 = set_x_n_1[:, 0]
    y_2 = set_x_n_1[:, 1]
    z_2 = set_x_n_1[:, 2]
    # box particles whose 2nd dimension is set to be 1
    set_y_1 = np.array(set_y_1)
    x_3 = set_y_1[:, 0]
    y_3 = set_y_1[:, 1]
    z_3 = set_y_1[:, 2]
    # box particles whose 2nd dimension is set to be -1
    set_y_n_1 = np.array(set_y_n_1)
    x_4 = set_y_n_1[:, 0]
    y_4 = set_y_n_1[:, 1]
    z_4 = set_y_n_1[:, 2]
    # box particles whose 3rd dimension is set to be 1
    set_z_1 = np.array(set_z_1)
    x_5 = set_z_1[:, 0]
    y_5 = set_z_1[:, 1]
    z_5 = set_z_1[:, 2]
    # box particles whose 3rd dimension is set to be -1
    set_z_n_1 = np.array(set_z_n_1)
    x_6 = set_z_n_1[:, 0]
    y_6 = set_z_n_1[:, 1]
    z_6 = set_z_n_1[:, 2]

    # visulization
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_1, y_1, z_1, color='lightcoral')   # [1, .., ..]
    ax.scatter(x_3, y_3, z_3, color='orange')   # [.., 1, ..]
    ax.scatter(x_5, y_5, z_5, color='yellow')   # [.., .., 1]
    ax.scatter(x_2, y_2, z_2, color='lime')     # [-1, .., ..]
    ax.scatter(x_4, y_4, z_4, color='cyan')     # [.., -1, ..]
    ax.scatter(x_6, y_6, z_6, color='violet')   # [.., .., -1]

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.interactive(True)
    plt.show()


def fluid_init_analysis(fluid_pos, fluid_vel):
    # analyse the range of initialized position and vel, origin data has 1000 particles as initialization
    x_pos_min, x_pos_max, x_pos_sum = 100, -100, 0
    y_pos_min, y_pos_max, y_pos_sum = 100, -100, 0
    z_pos_min, z_pos_max, z_pos_sum = 100, -100, 0
    with open(fluid_pos, 'r') as f_pos:
        for each_line in range(1000):
            data = f_pos.readline().split(',')
            pos_data = list(map(lambda x: float(x), data))

            if pos_data[0] < x_pos_min:
                x_pos_min = pos_data[0]
            if pos_data[0] > x_pos_max:
                x_pos_max = pos_data[0]
            x_pos_sum += pos_data[0]

            if pos_data[1] < y_pos_min:
                y_pos_min = pos_data[1]
            if pos_data[1] > y_pos_max:
                y_pos_max = pos_data[1]
            y_pos_sum += pos_data[1]

            if pos_data[2] < z_pos_min:
                z_pos_min = pos_data[2]
            if pos_data[2] > z_pos_max:
                z_pos_max = pos_data[2]
            z_pos_sum += pos_data[2]

    # calculate the range of each dimension
    x_mean = x_pos_sum / 1000
    y_mean = y_pos_sum / 1000
    z_mean = z_pos_sum / 1000

    print('positions situation: ')
    print('x_max: ', x_pos_max, 'x_min: ', x_pos_min, 'x_mean: ', x_mean)
    print('y_max: ', y_pos_max, 'y_min: ', y_pos_min, 'y_mean: ', y_mean)
    print('z_max: ', z_pos_max, 'z_min: ', z_pos_min, 'z_mean: ', z_mean)

    x_vel_min, x_vel_max, x_vel_sum = 100, -100, 0
    y_vel_min, y_vel_max, y_vel_sum = 100, -100, 0
    z_vel_min, z_vel_max, z_vel_sum = 100, -100, 0
    with open(fluid_vel, 'r') as f_vel:
        for each_line in range(1000):
            data = f_vel.readline().split(',')
            vel_data = list(map(lambda x: float(x), data))

            if vel_data[0] < x_vel_min:
                x_vel_min = vel_data[0]
            if vel_data[0] > x_vel_max:
                x_vel_max = vel_data[0]
            x_vel_sum += vel_data[0]

            if vel_data[1] < y_vel_min:
                y_vel_min = vel_data[1]
            if vel_data[1] > y_vel_max:
                y_vel_max = vel_data[1]
            y_vel_sum += vel_data[1]

            if vel_data[2] < z_vel_min:
                z_vel_min = vel_data[2]
            if vel_data[2] > z_vel_max:
                z_vel_max = vel_data[2]
            z_vel_sum += vel_data[2]

    # calculate the range of each dimension
    x_vel_mean = x_vel_sum / 1000
    y_vel_mean = y_vel_sum / 1000
    z_vel_mean = z_vel_sum / 1000

    print('velocity situation: ')
    print('x_max: ', x_vel_max, 'x_min: ', x_vel_min, 'x_mean: ', x_vel_mean)
    print('y_max: ', y_vel_max, 'y_min: ', y_vel_min, 'y_mean: ', y_vel_mean)
    print('z_max: ', z_vel_max, 'z_min: ', z_vel_min, 'z_mean: ', z_vel_mean)


def fluid_range_analysis(fluid_pos, fluid_vel):
    # analyse the range of position and vel in the whole process
    x_pos_min, x_pos_max, x_pos_sum = 100, -100, 0
    y_pos_min, y_pos_max, y_pos_sum = 100, -100, 0
    z_pos_min, z_pos_max, z_pos_sum = 100, -100, 0
    count = 0
    with open(fluid_pos, 'r') as f_pos:
        for each_line in f_pos:
            count += 1
            data = each_line.split(',')
            pos_data = list(map(lambda x: float(x), data))

            if pos_data[0] < x_pos_min:
                x_pos_min = pos_data[0]
            if pos_data[0] > x_pos_max:
                x_pos_max = pos_data[0]
            x_pos_sum += pos_data[0]

            if pos_data[1] < y_pos_min:
                y_pos_min = pos_data[1]
            if pos_data[1] > y_pos_max:
                y_pos_max = pos_data[1]
            y_pos_sum += pos_data[1]

            if pos_data[2] < z_pos_min:
                z_pos_min = pos_data[2]
            if pos_data[2] > z_pos_max:
                z_pos_max = pos_data[2]
            z_pos_sum += pos_data[2]

    # calculate the range of each dimension
    x_mean = x_pos_sum / count
    y_mean = y_pos_sum / count
    z_mean = z_pos_sum / count

    print('positions situation: ')
    print('x_max: ', x_pos_max, 'x_min: ', x_pos_min, 'x_mean: ', x_mean)
    print('y_max: ', y_pos_max, 'y_min: ', y_pos_min, 'y_mean: ', y_mean)
    print('z_max: ', z_pos_max, 'z_min: ', z_pos_min, 'z_mean: ', z_mean)

    x_vel_min, x_vel_max, x_vel_sum = 100, -100, 0
    y_vel_min, y_vel_max, y_vel_sum = 100, -100, 0
    z_vel_min, z_vel_max, z_vel_sum = 100, -100, 0
    count = 0
    with open(fluid_vel, 'r') as f_vel:
        for each_line in f_vel:
            count += 1
            data = each_line.split(',')
            vel_data = list(map(lambda x: float(x), data))

            if vel_data[0] < x_vel_min:
                x_vel_min = vel_data[0]
            if vel_data[0] > x_vel_max:
                x_vel_max = vel_data[0]
            x_vel_sum += vel_data[0]

            if vel_data[1] < y_vel_min:
                y_vel_min = vel_data[1]
            if vel_data[1] > y_vel_max:
                y_vel_max = vel_data[1]
            y_vel_sum += vel_data[1]

            if vel_data[2] < z_vel_min:
                z_vel_min = vel_data[2]
            if vel_data[2] > z_vel_max:
                z_vel_max = vel_data[2]
            z_vel_sum += vel_data[2]

    # calculate the range of each dimension
    x_vel_mean = x_vel_sum / count
    y_vel_mean = y_vel_sum / count
    z_vel_mean = z_vel_sum / count

    print('velocity situation: ')
    print('x_max: ', x_vel_max, 'x_min: ', x_vel_min, 'x_mean: ', x_vel_mean)
    print('y_max: ', y_vel_max, 'y_min: ', y_vel_min, 'y_mean: ', y_vel_mean)
    print('z_max: ', z_vel_max, 'z_min: ', z_vel_min, 'z_mean: ', z_vel_mean)


def readable_fluids(fluids_file):
    # fluid_train has the form of [[pos_matrix, vel_matrix, label1_matrix, label2_matrix], [s..t...e..p..2], ...]
    with open(fluids_file, 'rb') as f:
        fluids = pickle.load(f)

    res = []
    count = 0
    for each_step in fluids:
        count += 1
        for each_pos in each_step[0]:
            if each_pos[1] < 0:
                print(count)
        res.append(np.hstack((each_step[0], each_step[1])))

    res = np.concatenate(res)
    np.savetxt('fluid_txt.txt', res)
    return




# get_particle_data('apic2d_data.txt', 'apic2d_data.pkl', 200)
# apic_2d_data_clean('apic2d_data.pkl', 'apic2d_data_clean.txt')
# normal_analysis('box_normals.out', 'box.out')

# fluid_init_analysis('fluid_pos.txt', 'fluid_vel.txt')
# pos_x, pos_z: [-0.5, 0.5], pos_y: [0.5, 1.5], vel_x and vel_z has a 0.2~0.3 small value

# fluid_range_analysis('fluid_pos.txt', 'fluid_vel.txt')
# data_normalization('apic2d_data.pkl', 'fluid_train.pkl')
# readable_fluids('fluid_train.pkl')
# boundary particles generation
# boundary_generation('box_train.pkl')
# boundary_generation_no_wall('box_train_no_wall.pkl')