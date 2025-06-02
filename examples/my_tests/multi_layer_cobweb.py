import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from cobweb.cobweb import CobwebTree
from cobweb.cobweb_torch import CobwebTorchTree
from cobweb.visualize import load_tree, visualize


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":

    #torch.manual_seed(1234)
    

    transform = transforms.Compose(
      [transforms.ToTensor()])
    trainSet = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    
    torch.manual_seed(20)
    data_loader = torch.utils.data.DataLoader(trainSet,
                                          #batch_size=len(trainSet),
                                          batch_size=10,
                                          shuffle=True,
                                          num_workers=3)
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                          #batch_size=len(test_set),
                                          batch_size=10,
                                          shuffle=True,
                                          num_workers=3)

    train_images, train_labels = next(iter(data_loader))
    # print("train_images:",train_images.shape)
    test_images, test_labels = next(iter(test_data_loader))
    # print("test_images:",test_images.shape)

    # # 1. Testing Cobweb/4V on whole MNIST images
    # #####################################
    

    
    # # 2. Testing Cobweb.cpp on the new representation of MNIST images from Cobweb/4V
    # ########################################################################
    # categorizing the testing patches into a pre-trained Cobweb tree to get the new representation
    # 8x8 Patches, MUCH SLOWER
    id = 50 # id of the saved cobweb tree
    num_patches = id * 400
    new_tree = CobwebTorchTree(train_images.shape[1:])
    loaded_tree = load_tree(new_tree,id)
    print("Cobweb tree trained on",id, "8x8 patches") 

    # # New representation for training images
    list_new_reps_Train = []
    for i in tqdm(range(train_images.shape[0])):
        new_represenations = {}
        label_dict = {}
        key1 = f"{train_labels[i]}"
        label_dict[key1] = 1
        new_represenations["digit_label"] = label_dict
        
        for x in range(train_images.shape[2] - 8):
            for y in range(train_images.shape[3] - 8):

                key2 = f"{x},{y}"
                leaf = loaded_tree.categorize(train_images[i, :, x:x+8, y:y+8])
                inner_dict = {} # for holding the path for each patch
                while leaf is not None:
                    c_id = leaf.concept_id
                    key3 = f"{c_id}"
                    inner_dict[key3] = 1
                    leaf = leaf.parent
                new_represenations[key2] = inner_dict
                
        list_new_reps_Train.append(new_represenations) # holding the paths for all patches in an image

    #print(list_new_reps_Train) 
    
    # # New representation for testing images
    list_new_reps_Test = []
    for i in tqdm(range(test_images.shape[0])):
        new_represenations = {}
        label_dict = {}
        key1 = f"{test_labels[i]}"
        label_dict[key1] = 1
        new_represenations["digit_label"] = label_dict
        
        for x in range(test_images.shape[2] - 8):
            for y in range(test_images.shape[3] - 8):

                key2 = f"{x},{y}"
                leaf = loaded_tree.categorize(test_images[i, :, x:x+8, y:y+8])
                inner_dict = {} # for holding the path for each patch
                while leaf is not None:
                    c_id = leaf.concept_id
                    key3 = f"{c_id}"
                    inner_dict[key3] = 1
                    leaf = leaf.parent
                new_represenations[key2] = inner_dict
                
        list_new_reps_Test.append(new_represenations) # holding the paths for all patches in an image
        
    #print(list_new_reps_Test) 
        

    tree = CobwebTree(0.001, False, 0, True, False)
    for instance in tqdm(list_new_reps_Train):
        tree.ifit(instance)
    
    n_correct = 0   
    concepts_pred = []
    for i in tqdm(range(len(list_new_reps_Test))):
        instance = list_new_reps_Test[i]
        actual_label = list(instance['digit_label'].keys())[0]
        print("actual label:",actual_label)
       
        probs_pred = tree.predict_probs(instance, 50, False, False, 1)
        #probs_pred = tree.categorize(instance).predict_probs()
        
        digit_dict = probs_pred['digit_label']
        predicted_label = max(digit_dict, key=digit_dict.get)
        print("predicted label:",predicted_label)
        # print("props of all labels:",probs_pred['digit_label'])
        # print("\n")
        if (predicted_label == actual_label):
            n_correct += 1

    Accuracy = (n_correct/len(list_new_reps_Test))*100
    print("Accuracy: ",Accuracy)

    visualize(tree)  

    












