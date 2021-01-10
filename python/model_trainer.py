import torch
from DETR_light import DETR
from cocoDataset import CocoDataset
from loss import loss, getBipartiteMatching
from train import train

device = torch.device("cuda")
model = DETR().to(device)

trainDataset = CocoDataset(100, "../data/annotations/cleaned_train_image_data.csv", "../data/annotations/cleaned_train_labels_data.csv", "../data/train2017")
valDataset = CocoDataset(100, "../data/annotations/cleaned_val_image_data.csv", "../data/annotations/cleaned_val_labels_data.csv", "../data/val2017")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

torch.cuda.empty_cache()
model = train(model, optimizer, loss, trainDataset, valDataset, device, epochs=1, batchSize=1)

torch.save(model, 'trained_DETR_testing_epochs_1.pkl')