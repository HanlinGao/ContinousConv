import torch
import numpy as np
import pickle
from sklearn import preprocessing


def get_particle_data(file, save_file, num_lines=10):
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

        # add labels, so for each time step in dataset, the form will be [pos_matrix, vel_matrix, label_matrix]
        for j in range(num_lines):
            dataset[j].append(dataset[j+1][0])
            dataset[j].append(dataset[j+2][0])

    # save into a .pkl file
    with open(save_file, 'wb') as f:
        pickle.dump(dataset[: -2], f)
    return


def getBoundary(save_file, origin=[20., 20.], width=60., height=60., gap=1):
    pos = []
    for x in np.arange(origin[0], origin[0]+width, gap):
        pos.append([float(x), float(origin[1]), 0.])
        pos.append([float(x), float(origin[1]+height), 0.])

    for y in np.arange(origin[1], origin[1]+height, gap):
        pos.append([float(origin[1]), float(y), 0.])
        pos.append([float(origin[1]+width), float(y), 0.])

    normals = [[0., 0., 0.]] * len(pos)
    box_data = [pos, normals]

    # save into a .pkl file
    with open(save_file, 'wb') as f:
        pickle.dump(box_data, f)
    return


def preprocess(data):
    # Create scaler
    scaler = preprocessing.StandardScaler()
    # Transform the feature
    features_standardized = scaler.fit_transform(data)
    return features_standardized


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):
        super().__init__()
        with open(datafile, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
