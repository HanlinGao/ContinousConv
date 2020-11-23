import torch
import pickle


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