import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


def get_dataLoader_mix(transformSequence, trans_aug, labelled=50, batch_size=8,
                       txtFilePath='Labels/',
                       pathDirData =  'data/ISIC2018_Task3_Training_Input/'):

    pathFileTrain_L =  txtFilePath + '/Train' + str(labelled) + '.txt'
    pathFileTrain_U =  txtFilePath + '/Train_unl' + str(labelled) + '.txt'
    validation =  txtFilePath + '/Val.txt'
    test =  txtFilePath + '/Test.txt'


    datasetTrainLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_L],
                                                    transform=trans_aug)
    datasetTrainUnLabelled = DatasetGenerator_Mix(path=pathDirData, textFile=[pathFileTrain_U],
                                                    transform=TransformTwice(trans_aug))
    datasetVal = DatasetGenerator_Mix(path=pathDirData, textFile=[validation],
                                                    transform=transformSequence)
    datasetTest = DatasetGenerator_Mix(path=pathDirData, textFile=[test],
                                                    transform=transformSequence)


    dataLoaderTrainLabelled = DataLoader(dataset=datasetTrainLabelled, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderTrainUnLabelled = DataLoader(dataset=datasetTrainUnLabelled, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return dataLoaderTrainLabelled, dataLoaderTrainUnLabelled, dataLoaderVal, dataLoaderTest

class DatasetGenerator_Mix(Dataset):
    def __init__(self, path, textFile, transform):
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

        pathDatasetFile = textFile[0]
        fileDescriptor = open(pathDatasetFile, "r")
        line = True

        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                nameSplit = lineItems[0]
                if 'test' in path:
                    imagePath = os.path.join(path, nameSplit)
                else:
                    imagePath = os.path.join(path, nameSplit)
                imageLabel = lineItems[1:]
                imageLabel = [int(float(i)) for i in imageLabel]

                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        # imageData = Image.open(imagePath).convert('L')
        imageData = Image.open(imagePath)
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)


if __name__ == '__main__':
    #Transforms for the data
    import torchvision.transforms as transforms
    #Transforms for the data
    transformSequence = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
            ])
    
    trans_aug = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomRotation(degrees=(-10,10)),
            transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor()
            ])
    
    labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(
        transformSequence, trans_aug, labelled=50, batch_size=1)

    for ii, (sample, label) in enumerate(labeled_trainloader):
        for jj in range(sample.size()[0]):
            print(sample[jj].shape)
            print(label[jj])
        if ii == 1000:
            break
