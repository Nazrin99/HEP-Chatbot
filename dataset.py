from torch.utils.data import Dataset

class HepChatbotDataset(Dataset):

    def __init__(self, x, y):
        self.n_samples = len(x)
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.n_samples