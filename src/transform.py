import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from constant import device


class GaussianNoise(object):
    """
    Class for random gaussian noise generation


    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        """Generation of random Gaussian noise to improve training stability"""
        return tensor + torch.randn(tensor.size(),device=device) * self.std + self.mean

class Rotate90(object):
    """This class implement custom discrete rotation by angles which are multiples of 90° degrees


    """
    def __call__(self, img):
        angle = torch.randint(0, 4, (1,), device=device) * 90  
        return TF.rotate(img, int(angle.item()))




class AugmentedDataset(torch.utils.data.Dataset):
    """
    Class for dataset augmentation: it allows to apply custom transformation

    Args:
        torch (Dataset): the class takes a tensor Dataset as input.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(torch.tensor(image, dtype=torch.float32))
        return image, label

