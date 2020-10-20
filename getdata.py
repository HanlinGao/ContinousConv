import math
import torch


def getDynamics(file, num_lines=10):

    dataset = []

    with open(file, "r") as f:
        for i in range(num_lines+1):
            line = f.readline()

            step = []
            pos = []
            vel = []

            line = line.rstrip().split(' | ')

            for i in range(len(line)):
                p = []
                v = []
                cur = line[i].split("\t")

                p.append(cur[0])
                p.append(cur[1])
                v.append(cur[2])
                v.append(cur[3])
                pos.append(p)
                vel.append(v)

            step.append(pos)
            step.append(vel)
            dataset.append(step)
        for j in range(num_lines):
            dataset[j].append(dataset[j+1][0])
    return dataset[:-1]


def getBoundary(dataset, origin=[0, 0], width=100, height=100, gap=2):
    numofsteps = len(dataset)
    pos = []

    for x in range(origin[0], origin[0]+width, gap):
        pos.append([x, origin[1]])
    for y in range(origin[1], origin[1]+height, gap):
        pos.append([origin[1], y])
        pos.append([origin[1]+width, y])


    vel = [[math.inf, math.inf]]*len(pos)
    step = [pos, vel, pos]
    b = [step]*numofsteps
    return b


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, num_lines):
        super().__init__()
        self.dynamics = getDynamics(file_path, num_lines)
        self.boundary = getBoundary(self.dynamics)

    def __getitem__(self, type, time_step):
        if str(type) == "d":
            data, gt = self.dynamics[time_step]
            return data, gt
        elif str(type) == "b":
            data, gt = self.boundary[time_step]
            return data, gt
        else:
            print("invalid type")
            return

    def __len__(self):
        return len(self.boundary)

    def tofile(self):

        with open('dynamics.txt', 'w') as f1:
            for item in self.dynamics:
                f1.write("%s\n" % item)
        with open('boundary.txt', 'w') as f2:
            for item in self.boundary:
                f2.write("%s\n" % item)

if __name__ == '__main__':
    dataset = MyDataset("output.txt", 100)
    dataset.tofile()




