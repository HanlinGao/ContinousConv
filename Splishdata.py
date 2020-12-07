import pysplishsplash as sph
import os
import pyvista as pv
import numpy as np
import pickle


def create_raw_data(dir_name, sceneFile):
    output_dir = os.path.join(os.getcwd(), dir_name)

    base = sph.Exec.SimulatorBase()
    base.init(sceneFile=sceneFile, outputDir=output_dir)

    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


def read_vtk(vtk_dir, save_file, mode):
    dataset = []
    vtk_dir_path = os.path.join(os.getcwd(), vtk_dir)
    for root, dirs, files in os.walk(vtk_dir_path):
        files.sort(key=lambda x: int(x[19:-4]))
        for name in files:
            mesh = pv.read(os.path.join(vtk_dir_path, name))
            pos_matrix = np.array(mesh.points)
            vel_matrix = np.array(mesh.point_arrays['velocity'])
            ids = np.array(mesh.point_arrays['id'])

            # particle is stored in different order in different time, so need ordering
            pos_matrix = np.insert(pos_matrix, 0, values=ids, axis=1)
            vel_matrix = np.insert(vel_matrix, 0, values=ids, axis=1)
            # print(pos_matrix)
            pos_matrix = pos_matrix[np.argsort(pos_matrix[:, 0])]
            vel_matrix = vel_matrix[np.argsort(vel_matrix[:, 0])]

            # print(pos_matrix)
            dataset.append([pos_matrix[:, 1:], vel_matrix[:, 1:]])

    # print(len(dataset))
    # add label, finally the dataset will be [pos_matrix, vel_matrix, pos_matrix_t+1, pos_matrix_t+2]
    for j in range(len(dataset)-2):
        dataset[j].append(dataset[j+1][0][:])
        dataset[j].append(dataset[j+2][0][:])

    # save into a .pkl file
    with open(save_file, mode) as f:
        pickle.dump(dataset[: -2], f)

    print(len(dataset))
    return


def boundary_generation(box_save_file):
    # generate [box_pos_matrix, box_normals_matrix]
    box_pos = []
    box_normals = []

    # generate the bottom and ceiling
    bottom_x = np.arange(-2, 2.05, 0.05)
    for each in bottom_x:
        box_pos.append([each, -2., 0.])
        box_normals.append([0., 1., 0.])

        box_pos.append([each, 2., 0.])
        box_normals.append([0., -1., 0.])

    # generate left and right
    y = np.arange(-1.95, 2., 0.05)
    for each in y:
        box_pos.append([-2., each, 0.])
        box_normals.append([1., 0., 0.])

        box_pos.append([2., each, 0.])
        box_normals.append([-1., 0., 0.])

    box_pos = np.array(box_pos)
    box_normals = np.array(box_normals)
    with open(box_save_file, 'wb') as f:
        pickle.dump([box_pos, box_normals], f)


def dataset_modify(dataset_path, save_path, replicate_times=9):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # check each mapping in the dataset ([pos, vel, label1, label2]), classify them into different phases
    last_vel = dataset[0][1]    # vel_matrix
    timesteps_replicated = []
    num_map = 0
    for each_map in dataset[1:]:
        num_map += 1
        for p_i in range(len(each_map[1])):     # p_i represents particle i
            if each_map[1][p_i][0] > 0 and last_vel[p_i][0] < 0:
                print('collision detected', num_map)
                timesteps_replicated.append(num_map)
                # print('last vel', last_vel)
                # print('current vel', each_map[1])
                break
        last_vel = each_map[1]
    print('before modifying', len(dataset))
    for each in timesteps_replicated:
        dataset.extend([dataset[each]] * replicate_times)
    print('after modifying', len(dataset))
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


# sceneFile = 'D:/1_Study/1_ClassMaterials\Master1\Research\Datasets/Sampling_2D.json'
# create_raw_data('3p_hori_-2', sceneFile)
# read_vtk('3p_hori_-2/vtk', '3p_hori_-2.pkl', 'wb')

sceneFile = 'D:/1_Study/1_ClassMaterials\Master1\Research\Datasets/10p.json'
create_raw_data('20p_hori_0', sceneFile)
read_vtk('20p_hori_0/vtk', '20p_hori_0.pkl', 'wb')
# boundary_generation('box_train_box.pkl')
#
# create_raw_data('3p_-10_-5')
# read_vtk('3p_-10_-5/vtk', 'fluid_train_3p_-10_-5.pkl', 'wb')

# create_raw_data('3p_-2_0_vertical')
# read_vtk('3p_-2_0_vertical/vtk', 'fluid_train_3p_-2_0.pkl', 'wb')
# dataset_modify('fluid_train_3p_-5_-5.pkl', '3p_-5_-5_modified.pkl')