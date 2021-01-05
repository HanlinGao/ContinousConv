import torch
import argparse
from Model import MyParticleNetwork
import numpy as np
import sys
import pickle


def run_sim_torch(box_file, fluids_file, output_file, weight_path, num_step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init the network
    model = MyParticleNetwork()
    weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.to(device)
    model.requires_grad_(False)

    # init box and dynamic particles
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)
        box = box_data[0]
        box_normals = box_data[1]
    with open(fluids_file, 'rb') as f:
        particle_data = pickle.load(f)
        points = particle_data[0][0]
        velocities = particle_data[0][1]

    # compute lowest point for removing out of bounds particles
    min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))

    box = torch.from_numpy(box).float().to(device)
    box_normals = torch.from_numpy(box_normals).float().to(device)
    fluids = [(points, velocities, range(0, 1))]

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    print(len(box))
    print(box[:100])

    with open(output_file, 'wb') as f:
        for step in range(num_step):
            for point, velocities, range_ in fluids:
                if step in range_:  # check if we have to add the fluid at this point in time
                    pos = np.concatenate([pos, points], axis=0)
                    vel = np.concatenate([vel, velocities], axis=0)

            if pos.shape[0]:
                print('save', pos)
                np.save(f, pos)
                inputs = (torch.from_numpy(pos).float().to(device),
                          torch.from_numpy(vel).float().to(device), None, box, box_normals)

                pos, vel = model(inputs)
                pos = pos.cpu().numpy()
                vel = vel.cpu().numpy()
                pos[:, -1] = 0.0
                vel[:, -1] = 0.0

            # # remove out of bounds particles
            # if step % 10 == 0:
            #     print(step, 'num particles', pos.shape[0])
            #     mask = pos[:, 1] > min_y
            #     if np.count_nonzero(mask) < pos.shape[0]:
            #         pos = pos[mask]
            #         vel = vel[mask]


def main():
    parser = argparse.ArgumentParser(
        description="Run the network to predict multiple time steps, input box and particle information")
    parser.add_argument("--weights",
                        type=str,
                        required=True,
                        help="The path to the .pt weights file for torch.")
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--box_file",
                        type=str,
                        required=True,
                        help="A numpy file which describes the box positions and box normals")
    parser.add_argument("--particle_file",
                        type=str,
                        required=True,
                        help="A numpy file which describes the dynamic particle positions and features")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")

    args = parser.parse_args()
    print(args)

    return run_sim_torch(args.box_file, args.particle_file, args.output, args.weights, args.num_steps)

if __name__ == '__main__':
    sys.exit(main())