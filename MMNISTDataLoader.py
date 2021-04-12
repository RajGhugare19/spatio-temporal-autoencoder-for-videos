import torch 

class MMNISTTrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, num_ids):
        
        self.num_IDs = num_ids

    def __getitem__(self, index):
        
        assert index<self.num_IDs
        x = torch.load('MMNIST/train/id-' + str(index) + '.pt')
        return x
    
    def __len__(self):

        return self.num_IDs

class MMNISTTestDataset(torch.utils.data.Dataset):

    def __init__(self, num_ids):
        
        self.num_IDs = num_ids

    def __getitem__(self, index):
        
        assert index<self.num_IDs
        x = torch.load('MMNIST/test/id-' + str(index) + '.pt')
        return x
    
    def __len__(self):

        return self.num_IDs