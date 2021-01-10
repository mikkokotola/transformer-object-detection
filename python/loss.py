import torch
import numpy as np
from scipy import sparse

IoUWeight = 0.8


def loss(predictedClasses, predictedBoxes, classes, boxes):
    matching = getBipartiteMatching(
        predictedClasses, predictedBoxes, classes, boxes)
    classes = torch.stack([classes[i, matching[i], :]
                           for i in range(len(matching))], dim=0)
    boxes = torch.stack([boxes[i, matching[i], :]
                         for i in range(len(matching))], dim=0)
    return torch.mean(getMatchLoss(predictedClasses, predictedBoxes, classes, boxes))


def getMatchLoss(class1, box1, class2, box2):
    classLoss = getClassLoss(class1, class2)
    hasClass = torch.argmax(class1, dim=2) != 0
    boxLoss = IoUWeight * \
        getBoxGIoULoss(box1, box2) + (1-IoUWeight)*getBoxL1Loss(box1, box2)
    boxLoss = boxLoss*hasClass
    return classLoss + boxLoss


def getClassLoss(class1, class2):
    class1 = 0.999*class1 + 0.0005
    return -torch.sum(class2*torch.log(class1), dim=2)


def getBoxL1Loss(box1, box2):
    return torch.sum(torch.abs(box1-box2), dim=2)


def getBoxGIoULoss(box1, box2):

    box1left, box1right, box1top, box1bottom = getBoxBoundaries(box1)
    box2left, box2right, box2top, box2bottom = getBoxBoundaries(box2)

    box1Area = (box1right - box1left) * (box1bottom - box1top)
    box2Area = (box2right - box2left) * (box2bottom - box2top)

    intersectionLeft = tensorMaximum(box1left, box2left)
    intersectionRight = tensorMinimum(box1right, box2right)
    intersectionTop = tensorMaximum(box1top, box2top)
    intersectionBottom = tensorMinimum(box1bottom, box2bottom)

    intersectionValid = torch.logical_and(
        intersectionRight > intersectionLeft, intersectionBottom > intersectionTop)
    intersectionArea = (intersectionRight - intersectionLeft) * \
        (intersectionBottom - intersectionTop)
    intersectionArea = intersectionArea * intersectionValid

    smallestEnclosingLeft = tensorMinimum(box1left, box2left)
    smallestEnclosingRight = tensorMaximum(box1right, box2right)
    smallestEnclosingTop = tensorMinimum(box1top, box2top)
    smallestEnclosingBottom = tensorMaximum(box1bottom, box2bottom)

    smallestEnclosingArea = (smallestEnclosingRight - smallestEnclosingLeft) * \
        (smallestEnclosingBottom - smallestEnclosingTop)

    unionArea = box1Area + box2Area - intersectionArea

    IoU = intersectionArea/unionArea

    GIoU = IoU - (smallestEnclosingArea - unionArea)/smallestEnclosingArea

    return 1 - GIoU

def getBoxBoundaries(box):
    boxleft = box[:, :, 0]-(box[:, :, 2]/2)
    boxright = box[:, :, 0]+(box[:, :, 2]/2)
    boxtop = box[:, :, 1]-(box[:, :, 3]/2)
    boxbottom = box[:, :, 1]+(box[:, :, 3]/2)
    return boxleft, boxright, boxtop, boxbottom


def getBipartiteMatching(predictedClasses, predictedBoxes, classes, boxes):
    predictedClasses = predictedClasses.detach().cpu()
    predictedBoxes = predictedBoxes.detach().cpu()
    classes = classes.detach().cpu()
    boxes = boxes.detach().cpu()

    graph = torch.stack([getMatchLoss(predictedClasses, predictedBoxes, torch.repeat_interleave(classes[:, i:(i+1), :], classes.shape[1],
                                                                                                dim=1), torch.repeat_interleave(boxes[:, i:(i+1), :], boxes.shape[1], dim=1)) for i in range(classes.shape[1])], dim=2)
    matching = [sparse.csgraph.min_weight_full_bipartite_matching(
        sparse.csr_matrix(graph[i, :, :].numpy()))[1] for i in range(graph.shape[0])]
    return matching

def tensorMaximum(t1, t2):
    maxVal = (t1 >= t2)*t1 + (t2 > t1)*t2
    return maxVal

def tensorMinimum(t1, t2):
    minVal = (t1 >= t2)*t2 + (t2 > t1)*t1
    return minVal
