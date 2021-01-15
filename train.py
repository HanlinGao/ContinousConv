#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from net import MyParticleNetwork
import matplotlib.pyplot as plt
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import datetime
from tensorboardX import SummaryWriter


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, device):
        super().__init__()
        self.device = device
        self.dataset = []
        with open(datafile, 'rb') as f:
            while True:
                try:
                    self.dataset.extend(pickle.load(f))
                except:
                    break

    def __getitem__(self, idx):
        pos_matrix = torch.from_numpy(self.dataset[idx][0]).float().to(self.device)
        vel_matrix = torch.from_numpy(self.dataset[idx][1]).float().to(self.device)
        label1 = torch.from_numpy(self.dataset[idx][2]).float().to(self.device)
        label2 = torch.from_numpy(self.dataset[idx][3]).float().to(self.device)

        return pos_matrix, vel_matrix, label1, label2

    def __len__(self):
        return len(self.dataset)


def create_model():
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork()
    return model


def euclidean_distance(a, b, epsilon=1e-9):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + epsilon)


def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gamma = 0.5
    neighbor_scale = 1 #TODO
    importance = torch.exp(-neighbor_scale * num_fluid_neighbors).to(device)
    return torch.mean(importance *
                      euclidean_distance(pr_pos, gt_pos) ** gamma)


def train(model, optimizer, batch, box_data):
    optimizer.zero_grad()
    losses = []
    for batch_i in range(len(batch[0])):
        inputs = ([
            batch[0][batch_i], batch[1][batch_i], None, box_data[0], box_data[1]
        ])

        pr_pos1, pr_vel1 = model(inputs)

        # force 3rd dimension to zero.
        # pr_pos1[:, -1] = 0.
        # pr_vel1[:, -1] = 0.
        # print("pr-pos1 modify: " + str(pr_pos1))

        l = 0.5 * loss_fn(pr_pos1, batch[2][batch_i], model.num_fluid_neighbors)

        inputs = (pr_pos1, pr_vel1, None, box_data[0], box_data[1])
        pr_pos2, pr_vel2 = model(inputs)

        # force 3rd dimension to zero.
        # pr_pos2[:, -1] = 0.
        # pr_vel2[:, -1] = 0.
        l += 0.5 * loss_fn(pr_pos2, batch[3][batch_i], model.num_fluid_neighbors)

        losses.append(l)

    total_loss = 128 * sum(losses) / len(batch[0])
    total_loss.backward()
    optimizer.step()

    return total_loss


def validate(model, valset, box_data):
    with torch.no_grad():
        losses = []
        for batch_i in range(len(valset)):
            inputs = ([
                valset[batch_i][0], valset[batch_i][1], None, box_data[0], box_data[1]
            ])

            pr_pos1, pr_vel1 = model(inputs)
            # pr_pos1[:, -1] = 0.
            # pr_vel1[:, -1] = 0.
            l = 0.5 * loss_fn(pr_pos1, valset[batch_i][2], model.num_fluid_neighbors)

            inputs = (pr_pos1, pr_vel1, None, box_data[0], box_data[1])
            pr_pos2, pr_vel2 = model(inputs)
            # pr_pos2[:, -1] = 0.
            # pr_vel2[:, -1] = 0.
            l += 0.5 * loss_fn(pr_pos2, valset[batch_i][3], model.num_fluid_neighbors)

            losses.append(l)

        total_loss = 128 * sum(losses) / len(valset)
    # print('validate: ', valset[0])
    return total_loss


def main(args):
    # Create model directory for saving trained models
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    # dataset = MyDataset(os.path.join(args.dataset_path, args.train_set + '.txt'), 200).dataset
    # testset = MyDataset(os.path.join(args.dataset_path, args.validate_set + '.txt'), 200).dataset

    print('data preparing....')
    dataset = MyDataset(os.path.join(args.dataset_path, args.train_set + '.pkl'), device)
    valset = MyDataset(os.path.join(args.dataset_path, args.validate_set + '.pkl'), device)

    # print('dataset[0]', dataset[0][0])
    print('dataset length: ', len(dataset))
    validate_data = []
    for i in range(0, args.batch_size):
        validate_data.append(valset[i])

    with open(os.path.join(args.dataset_path, args.box_data + '.pkl'), 'rb') as f:
        box_data = pickle.load(f)  # include box_pos, box_normals

    box_data = [torch.from_numpy(x).float().to(device) for x in box_data]

    # define model and optimizer
    model = create_model()
    model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(),
#                                  lr=args.lr,
#                                  eps=1e-6)
#TODO
    optimizer = swats.SWATS(model.parameters())
    # if os.path.isfile(os.path.join(args.model_path, args.model_name + '.pt')):
    #     print("load model " + args.model_path + args.model_name + '.pt')
    # model.load_state_dict(torch.load(os.path.join(args.model_path, args.model_name + '.pt')))
    model.load_state_dict(torch.load("model/pretrained_model_weights.pt"))

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(args.model_path, args.model_name + '.pt'))

    # get batches and steps
    # batches = toBatch(dataset, args.batch_size, device)
    # validates = toBatch(valset, args.batch_size, device)
    # ExpLr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    batches = len(dataset) // args.batch_size + 1

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    print('Done. Start training.')
    # total_batches = len(batches)

    epoch_tr = []
    epoch_val = []

    # start training
    for epoch in range(args.num_epochs):
        train_l = []
        validate_l = []
        train_iter = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True))
        for num_batch in range(batches):
            try:
                poses, vels, label1s, label2s = next(train_iter)
                batch = (poses, vels, label1s, label2s)
                current_loss = train(model, optimizer, batch, box_data)
                validate_loss = validate(model, validate_data, box_data)
                train_l.append(float(current_loss))
                validate_l.append(float(validate_loss))

            except StopIteration:
                break

        # ExpLr.step()
        epoch_tr.append(sum(train_l) / batches)
        epoch_val.append(sum(validate_l) / batches)

        writer.add_scalars('Loss-3p-x-04', {'train': sum(train_l)/batches,
                                        'val': sum(validate_l)/batches}, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        print('Epoch: {} /Loss: {} /val Loss: {}'.format(epoch, sum(train_l)/batches, sum(validate_l)/batches))
        # print('Epoch: {} /Loss: {}'.format(epoch, current_loss))

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model

        # early_stopping(validate_loss, model)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

        if '_' in args.model_name:
            index = args.model_name.find('_')
            model_name = args.model_name[:index]
        else:
            model_name = args.model_name
        torch.save(model.state_dict(), os.path.join(args.model_path, model_name + '_epoch_' + str(epoch) +
                                                    '_lr_' + str(args.lr) + '.pt'))
        print("Saving model...")

    # loss plot
    plt.plot(np.arange(1, len(epoch_tr)+1, 1), epoch_tr, "blue")
    plt.plot(np.arange(1, len(epoch_val)+1, 1), epoch_val, "red")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.savefig("train"+str(datetime.date.today().month) + str(datetime.date.today().day) + 'epoch_' + str(args.num_epochs) + '_lr_' + str(args.lr) + '.png')
    print("Loss plot saved")
    print("Finished, model saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_swats/', help='path for saving trained models')
    parser.add_argument('--dataset_path', type=str, default='dataset/input_train', help='apic2d dataset')
    parser.add_argument('--train_set', type=str, default='3p-middle', help='path for train set')
    parser.add_argument('--box_data', type=str, default='entire_box', help='boundary')
    parser.add_argument('--validate_set', type=str, default='eval', help='path for validate set')
    # parser.add_argument('--time_step', type=str, default=200, help='nums of time step')
    # Model parameters
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_name', type=str, default=str(datetime.date.today().month) + str(datetime.date.today().day))
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
