import torch
import pandas as pd
from PIL import Image
import os
import math
import torchvision


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, numberOfQueries, imagesTablePath, labelsTablePath, imagesPath):
        super(CocoDataset, self).__init__()
        self.imagesTable = pd.read_csv(imagesTablePath)
        self.labelsTable = pd.read_csv(labelsTablePath)
        self.imagesPath = imagesPath
        self.numberOfQueries = numberOfQueries

    def __len__(self):
        return len(self.imagesTable)

    def __getitem__(self, index):
        imageTensor = self.processImage(index)
        labels, boxes = self.processLabels(index)
        return imageTensor, labels, boxes

    def processImage(self, index):
        imageSeries = self.imagesTable.iloc[index]
        image = Image.open(os.path.join(
            self.imagesPath, imageSeries["file_name"])).convert("RGB")
        imageResized = image.resize(
            (imageSeries["scaled_width"], imageSeries["scaled_height"]), Image.LANCZOS)

        imageTensor = torchvision.transforms.ToTensor()(imageResized)
        imageTensor = torch.nn.functional.pad(
            imageTensor, (imageSeries["padding_left"], imageSeries["padding_right"], imageSeries["padding_top"], imageSeries["padding_bottom"]))

        return imageTensor

    def processLabels(self, index):
        imageSeries = self.imagesTable.iloc[index]
        labelsTable = self.labelsTable[self.labelsTable["image_id"]
                                       == imageSeries["id"]]
        classes = torch.zeros((len(labelsTable), 92), dtype=torch.float)
        for i in range(len(labelsTable)):
            classes[i, labelsTable.iloc[i]["category_id"]] = 1
        boxes = torch.tensor(labelsTable[["normalized_box_center_x", "normalized_box_center_y",
                                          "normalized_box_width", "normalized_box_height"]].values, dtype=torch.float)
        nonClasses = torch.zeros((100-classes.shape[0], 92))
        nonClasses[:, 0] = 1
        classes = torch.cat((classes, nonClasses), dim=0)

        nonBoxes = torch.ones((100-boxes.shape[0], 4))
        boxes = torch.cat((boxes, nonBoxes), dim=0)
        return classes, boxes
