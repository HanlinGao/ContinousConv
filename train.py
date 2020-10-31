#!/usr/bin/env python3
import numpy as np
import torch
from Model import MyParticleNetwork
from dataprocess import MyDataset
import matplotlib.pyplot as plt
import pickle


def main(box_file, particles_file, eval_file, lr=0.0125):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyDataset(particles_file)
    eval_dataset = MyDataset(eval_file)
    data_iter = iter(dataset)   # each time step: positions, vels, labels(next_pos)
    with open(box_file, 'rb') as f:
        box_data = pickle.load(f)   # include box_pos, box_normals
        box_pos = None
        box_normals = None

    model = MyParticleNetwork()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001,
                                 eps=1e-6)

    def euclidean_distance(a, b, epsilon=1e-9):
        return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        # print(num_fluid_neighbors)
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        return torch.mean(importance *
                          euclidean_distance(pr_pos, gt_pos) ** gamma)

    def train(model, batch, box_pos, box_normals):
        optimizer.zero_grad()
        losses = []
        batch_size = 16
        for batch_i in range(batch_size):
            inputs = ([
                torch.tensor(batch[batch_i][0]), torch.tensor(batch[batch_i][1]), None,
                box_pos, box_normals
            ])

            pr_pos1, pr_vel1 = model(inputs)
            l = 0.5 * loss_fn(pr_pos1, torch.tensor(batch[batch_i][2]),
                              model.num_fluid_neighbors)

            inputs = (pr_pos1, pr_vel1, None, box_pos, box_normals)
            pr_pos2, pr_vel2 = model(inputs)

            l += 0.5 * loss_fn(pr_pos2, torch.tensor(batch[batch_i][2]),
                               model.num_fluid_neighbors)

            losses.append(l)

        total_loss = 128 * sum(losses) / batch_size
        total_loss.backward()
        optimizer.step()

        return total_loss

    # def test(model, batch):
    #     losses = []
    #     batch_size = 16
    #     for batch_i in range(batch_size):
    #         inputs = ([
    #             torch.tensor(batch[batch_i][0]), torch.tensor(batch[batch_i][1]), None,
    #             torch.tensor(batch[batch_i][3]), torch.tensor(batch[batch_i][4])
    #         ])
    #
    #         pr_pos1, pr_vel1 = model(inputs)
    #         l = 0.5 * loss_fn(pr_pos1, torch.tensor(batch[batch_i][2]),
    #                           model.num_fluid_neighbors)
    #
    #         inputs = (pr_pos1, pr_vel1, None, box, torch.tensor(batch[batch_i][4]))
    #         pr_pos2, pr_vel2 = model(inputs)
    #
    #         l += 0.5 * loss_fn(pr_pos2, torch.tensor(batch[batch_i][5]),
    #                            model.num_fluid_neighbors)
    #
    #         losses.append(l)
    #
    #     total_loss = 128 * sum(losses) / batch_size
    #
    #     return total_loss

    def toBatch(data, batchsize):
        l = len(data)
        return np.asarray(np.array_split(data, l / batchsize))

    batches = toBatch(dataset, 16)
    # tests = toBatch(testset, 16)

    total_step = len(batches)
    train_l = []
    # test_l = []

    for epoch in range(100):
        for i in range(total_step):
            current_loss = train(model, batches[i], box_pos, box_normals)
            # test_loss = test(model, tests[i])
        train_l.append(current_loss)
        # test_l.append(test_loss)

        # print('Epoch: {} /Loss: {} /test Loss: {}'.format(epoch, current_loss, test_loss))

    plt.plot(np.arange(1, 101, 1), train_l, "blue")
    # plt.plot(np.arange(1, 101, 1), test_l, "red")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    torch.save({'model': model.state_dict()}, 'model_weights_lr_' + str(lr) + '.pt')


if __name__ == '__main__':
    main(box_file=, particles_file=, eval_file=, lr=)
