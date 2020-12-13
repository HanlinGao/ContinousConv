import pysplishsplash as sph
import os
import pyvista as pv
import numpy as np
import pickle
import json


def create_raw_data(dir_name, sceneFile):
    output_dir = os.path.join(os.getcwd(), dir_name)

    base = sph.Exec.SimulatorBase()
    base.init(sceneFile=sceneFile, outputDir=output_dir)

    gui = sph.GUI.Simulator_GUI_imgui(base)
    base.setGui(gui)
    base.run()


def read_vtk(vtk_dir, save_file, mode):
    dataset = []
    for root, dirs, files in os.walk(vtk_dir):
        files.sort(key=lambda x: int(x[19:-4]))
        for name in files:
            mesh = pv.read(os.path.join(vtk_dir, name))
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
    bottom_x = np.concatenate((np.arange(-2, 0, 0.05), np.arange(0, 2.05, 0.05)))
    for each in bottom_x:
        box_pos.append([each, -2., 0.])
        box_normals.append([0., 1., 0.])

        box_pos.append([each, 2., 0.])
        box_normals.append([0., -1., 0.])

    # generate left and right
    y = np.concatenate((np.arange(-1.95, 0, 0.05), np.arange(0, 2, 0.05)))
    for each in y:
        box_pos.append([-2., each, 0.])
        box_normals.append([1., 0., 0.])

        box_pos.append([2., each, 0.])
        box_normals.append([-1., 0., 0.])

    # generate front and back
    xv, yv = np.meshgrid(bottom_x, bottom_x)
    for i in zip(xv.flat, yv.flat):
        box_pos.append([i[0], i[1], 0.05])
        box_normals.append([0., 0., -1.])

        box_pos.append([i[0], i[1], -0.05])
        box_normals.append([0., 0., 1.])

    box_pos = np.array(box_pos)
    box_normals = np.array(box_normals)
    with open(box_save_file, 'wb') as f:
        pickle.dump([box_pos, box_normals], f)


def create_train_data(jsonfiles, save_file):
    root_path = os.getcwd()
    sceneFile = os.path.join(root_path, jsonfiles)
    train_dir = os.path.join(root_path, save_file)

    for root, ds, fs in os.walk(sceneFile):
        for scene in fs:
            full_name = os.path.join(sceneFile, scene)
            dir_name = os.path.splitext(scene)[0]

            vtk_save = os.path.join(root_path + '/VTK', dir_name)
            create_raw_data(vtk_save, full_name)
            read_vtk(vtk_save + '/vtk', os.path.join(train_dir, dir_name + '.pkl'), 'wb')


def create_jsons(configuration_file, save_dir, json_template='D:/1_Study/1_ClassMaterials/Master1/Research/Datasets/Sampling_2D.json'):
    root_path = 'D:/1_Study/1_ClassMaterials/Master1/Research/Datasets'
    json_save = os.path.join(root_path, save_dir)

    with open(json_template, 'r') as f:
        info = json.load(f)

    with open(configuration_file, 'r') as f:
        data_to_generate = json.load(f)

    for each_group in data_to_generate:
        print('each_group', each_group)
        count = 0
        for setting in data_to_generate[each_group]:
            print('setting', setting)
            info['FluidBlocks'][0]['start'] = setting['start']
            info['FluidBlocks'][0]['end'] = setting['end']
            info['FluidBlocks'][0]['initialVelocity'] = setting['initialVelocity']

            json_name = each_group + '_' + str(count) + '.json'
            with open(os.path.join(json_save, json_name), 'w') as fw:
                json.dump(info, fw)
            count += 1


def dataset_integrate(dataset_dir, save_trainset):
    for root, ds, fs in os.walk(os.path.join(os.getcwd(), dataset_dir)):
        dataset = []
        for pkl in fs:
            with open(os.path.join(root, pkl), 'rb') as f:
                dataset.extend(pickle.load(f))
        print(len(dataset))

        # save into a .pkl file
        with open(os.path.join(root, save_trainset), 'wb') as fw:
            pickle.dump(dataset, fw)


def eval_set(json_file, save_file):
    root = os.getcwd()
    vtkfile = os.path.join(root + '/VTK', 'eval')
    create_raw_data(vtkfile, os.path.join(root, json_file))
    read_vtk(vtkfile + '/vtk', os.path.join(root, save_file), 'wb')


# create_jsons('Jsonfiles/3p')
# create_train_data('Jsonfiles/3p', '3p_trainset')
# dataset_integrate('3p_trainset', '3p_trainset.pkl')
# eval_set('eval.json', 'eval.pkl')
# create_jsons('data_setting.json', 'Jsonfiles/3p_falling')
# create_train_data('Jsonfiles/3p_falling', '3p_falling')
# dataset_integrate('3p_falling', '3p_falling.pkl')
# eval_set('eval.json', 'eval.pkl')

boundary_generation('box_train_4wall.pkl')