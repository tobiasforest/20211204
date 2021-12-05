# A reproduction using PyTorch on the paper:
# UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS
# https://hal-enpc.archives-ouvertes.fr/hal-01864755

from Dataset import *
from torchvision import models

from torch.cuda import init
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image
from tqdm import tqdm

import copy
import time
import os


def train_model(model, criterion, loaders, optimizer, scheduler, num_epochs=NUM_EPOCHS, device='cuda'):
    """ Train the model """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            n_samples = 0

            with tqdm(total=len(loaders[phase])) as pbar:
                for i, (inputs, labels) in enumerate(loaders[phase]):
                    # Reshaping the inputs and labels
                    shapeList = list(inputs.size())[1:]
                    shapeList[0] = -1
                    inputs = torch.reshape(inputs, shapeList)
                    labels = torch.reshape(labels, (-1,))

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    n_samples += inputs.size(0)

                    pbar.set_description(
                        'Epoch {} / {}, Phase: {}'.format(epoch, num_epochs - 1, phase.capitalize()))
                    pbar.set_postfix(Loss='{:.4f}'.format(
                        running_loss/n_samples), Top_1_acc='{:.4f}'.format(running_corrects/n_samples))
                    pbar.update()

                end = time.time()

            # Metrics
            top_1_acc = running_corrects / n_samples
            epoch_loss = running_loss / n_samples

            # print('{} Loss: {:.4f} Top 1 Acc: {:.4f} Top k Acc: {:.4f}\n'.format(
            #     phase, epoch_loss, top_1_acc))

            # deep copy the model
            if phase == 'val' and top_1_acc > best_acc:
                best_acc = top_1_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), os.path.join(
                MODEL_SAVE_PATH, 'net_epoch_%d.pth' % epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet34(pretrained=False)
# model_ft = model_ft
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1000)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.Adam(model_ft.parameters(),
                            lr=0.001, betas=(BETA1, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

trainedModel = train_model(model=model_ft.to(DEVICE), criterion=criterion,
                           loaders=dataloaders, optimizer=optimizer_conv, scheduler=exp_lr_scheduler, device=DEVICE)
