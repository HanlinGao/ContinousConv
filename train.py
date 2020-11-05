#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from Model import MyParticleNetwork
import matplotlib.pyplot as plt
import os
import pickle


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):
        super().__init__()
        with open(datafile, 'rb') as f:
            self.dataset = pickle.load(f)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def create_model():
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork()
    return model


def euclidean_distance(a, b, epsilon=1e-9):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + epsilon)


def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
    gamma = 0.5
    neighbor_scale = 1 / 40
    importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
    return torch.mean(importance *
                      euclidean_distance(pr_pos, gt_pos) ** gamma)


def train(model, optimizer, batch, box_data, batch_size):
    optimizer.zero_grad()
    losses = []
    for batch_i in range(batch_size):
        inputs = ([
            torch.tensor(batch[batch_i][0], dtype=torch.float32), torch.tensor(batch[batch_i][1], dtype=torch.float32), None,
            torch.tensor(box_data[0], dtype=torch.float32), torch.tensor(box_data[1], dtype=torch.float32)
        ])

        pr_pos1, pr_vel1 = model(inputs)
        l = 0.5 * loss_fn(pr_pos1, torch.tensor(batch[batch_i][2], dtype=torch.float32),
                          model.num_fluid_neighbors)

        inputs = (pr_pos1, pr_vel1, None, torch.tensor(box_data[0], dtype=torch.float32), torch.tensor(box_data[1], dtype=torch.float32))
        pr_pos2, pr_vel2 = model(inputs)

        l += 0.5 * loss_fn(pr_pos2, torch.tensor(batch[batch_i][3], dtype=torch.float32),
                           model.num_fluid_neighbors)

        losses.append(l)

    total_loss = 128 * sum(losses) / batch_size
    total_loss.backward()
    optimizer.step()

    return total_loss


def validate(model, valset, box_data, batch_size):
    losses = []
    for batch_i in range(batch_size):
        inputs = ([
            torch.tensor(valset[batch_i][0], dtype=torch.float32), torch.tensor(valset[batch_i][1], dtype=torch.float32), None,
            torch.tensor(box_data[0], dtype=torch.float32), torch.tensor(box_data[1], dtype=torch.float32)
        ])

        pr_pos1, pr_vel1 = model(inputs)
        l = 0.5 * loss_fn(pr_pos1, torch.tensor(valset[batch_i][2], dtype=torch.float32),
                          model.num_fluid_neighbors)

        inputs = (pr_pos1, pr_vel1, None, torch.tensor(box_data[0], dtype=torch.float32), torch.tensor(box_data[1], dtype=torch.float32))
        pr_pos2, pr_vel2 = model(inputs)

        l += 0.5 * loss_fn(pr_pos2, torch.tensor(valset[batch_i][3], dtype=torch.float32),
                           model.num_fluid_neighbors)

        losses.append(l)

    total_loss = 128 * sum(losses) / batch_size

    return total_loss


def toBatch(data, batchsize):
    l = len(data)
    return np.asarray(np.array_split(data, l / batchsize))


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    # dataset = MyDataset(os.path.join(args.dataset_path, args.train_set + '.txt'), 200).dataset
    # testset = MyDataset(os.path.join(args.dataset_path, args.validate_set + '.txt'), 200).dataset

    dataset = MyDataset(os.path.join(args.dataset_path, args.train_set + '.pkl'))
    valset = MyDataset(os.path.join(args.dataset_path, args.validate_set + '.pkl'))

    with open(os.path.join(args.dataset_path, args.box_data + '.pkl'), 'rb') as f:
        box_data = pickle.load(f)  # include box_pos, box_normals

    # define model and optimizer
    model = create_model()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 eps=1e-6)

    if os.path.isfile(os.path.join(args.model_path, args.model_name + '.pt')):
        print("load model....")
        model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name + '.pt')))

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(args.model_path, args.model_name + '.pt'))

    # get batches and steps
    batches = toBatch(dataset, args.batch_size)
    validates = toBatch(valset, args.batch_size)

    total_step = len(batches)

    epoch_tr = []
    epoch_val = []

    # start training
    for epoch in range(args.num_epochs):
        train_l = []
        validate_l = []
        for i in range(total_step):
            # current_loss = train(model, optimizer, batches[i], args.batch_size)
            current_loss = train(model, optimizer, batches[i], box_data, args.batch_size)
            validate_loss = validate(model, validates[5], box_data, args.batch_size)

            train_l.append(current_loss)
            validate_l.append(validate_loss)

        epoch_tr.append(sum(train_l)/total_step)
        epoch_val.append(sum(validate_l) / total_step)

        print('Epoch: {} /Loss: {} /val Loss: {}'.format(epoch, sum(train_l)/total_step, sum(validate_l) / total_step))
        # print('Epoch: {} /Loss: {}'.format(epoch, current_loss))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

        # early_stopping(validate_loss, model)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        torch.save(model.state_dict(), os.path.join(args.model_path, args.model_name + '_epoch_' + str(epoch) + '.pt'))
        print("saving model...")

    # loss plot
    plt.plot(np.arange(1, len(train_l)+1, 1), train_l, "blue")
    plt.plot(np.arange(1, len(validate_l)+1, 1), validate_l, "red")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig('out/loss_new.png')
    print("loss plot saved")
    print("Finished, model saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='dataset', help='apic2d dataset')
    parser.add_argument('--train_set', type=str, default='fluid_train', help='path for train set')
    parser.add_argument('--box_data', type=str, default='box_train', help='boundary')
    parser.add_argument('--validate_set', type=str, default='validate', help='path for validate set')
    # parser.add_argument('--time_step', type=str, default=200, help='nums of time step')
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default="ourmodel")
    args = parser.parse_args()
    main(args)