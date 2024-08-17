import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from cobweb.cobweb_torch import CobwebTorchTree
from cobweb.visualize import save_tree, visualize


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose(
      [transforms.ToTensor()])

    ## MNIST
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)

    torch.manual_seed(20)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                          #batch_size= len(train_set),
                                          batch_size= 50,
                                          shuffle=True,
                                          num_workers=3)

    
    train_images, train_labels = next(iter(train_data_loader))
 
    # 8x8 Patches
    tree = CobwebTorchTree((1,8,8))
    for i in tqdm(range(train_images.shape[0])):
        for x in range(train_images.shape[2] - 8):
            for y in range(train_images.shape[3] - 8):
                tree.ifit(train_images[i, :, x:x+8, y:y+8])
        
        if ((i+1)%50 == 0):
            id = i+1
            print("checkpoint ",id)
            save_tree(tree,id)
        
    visualize(tree)
    # print("Loaded!")
    # visualize(loaded_tree)

