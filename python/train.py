import torch
import math
import time


def collate(data):
    images = torch.stack([item[0] for item in data], dim=0)
    classes = torch.stack([item[1] for item in data], dim=0)
    boxes = torch.stack([item[2] for item in data], dim=0)
    return images, classes, boxes


def train(model, optimizer, lossFunction, trainDataset, valDataset, device, epochs=50, batchSize=8):

    trainDataloader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=collate
    )

    valDataloader = torch.utils.data.DataLoader(
        dataset=valDataset,
        batch_size=batchSize,
        shuffle=True,
        collate_fn=collate
    )

    for epoch in range(epochs+1):

        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        print()

        phases = ["train", "val"] if epoch > 0 else ["val"]

        for phase in phases:

            print("Training..." if phase == "train" else "Validating...")
            print()

            dataloader = trainDataloader if phase == "train" else valDataloader

            if phase == "train":
                model.train()
            else:
                model.eval()

            lastSeenProgressProcent = -1
            runningLoss = 0
            numberOfPredictions = 0
            since = time.time()

            for index, samples in enumerate(dataloader):
                images, classes, boxes = samples

                images = images.to(device)
                classes = classes.to(device)
                boxes = boxes.to(device)

                numberOfPredictions += len(images)

                optimizer.zero_grad()
                torch.cuda.empty_cache()

                predictedClasses, predictedBoxes = model(images)

                loss = lossFunction(
                    predictedClasses, predictedBoxes, classes, boxes)
                loss.backward()

                runningLoss += loss.item()

                optimizer.step()

                lastSeenProgressProcent = printProgress(
                    index, batchSize, len(dataloader.dataset), lastSeenProgressProcent)

            print()
            print()

            stats = calculateEpochStats(
                runningLoss, numberOfPredictions, since)
            printStats(stats)

            print()

    return model


def printProgress(index, batchSize, datasetLength, lastSeenProgressProcent):
    progress = ((index+1)*batchSize)/(datasetLength)
    progressProcent = math.floor(progress * 100)

    if progressProcent >= lastSeenProgressProcent + 1:
        print("\rProgress: {}%".format(progressProcent), end="")
        lastSeenProgressProcent = progressProcent
    return lastSeenProgressProcent


def calculateEpochStats(loss, numberOfPredictions, since):
    epochTime = time.time() - since
    averageLoss = loss / numberOfPredictions

    return {
        "duration": epochTime,
        "loss": averageLoss,
    }


def printStats(stats):
    print('Loss: {:.4f} Duration: {:.0f}m {:.0f}s'.format(
        stats["loss"], stats["duration"] // 60, stats["duration"] % 60))
