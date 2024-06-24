import torch
from tqdm import tqdm

import os
from tempfile import TemporaryDirectory
import time


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader = None, num_epochs=5, device=None):
    since = time.time()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print('Using:', device)

    # do we have a validation set?
    has_val = val_loader is not None
    if has_val:
        phases = ['train', "val"] 
        dataloaders = dict(train=train_loader, val=val_loader)
    else:
        phases = ['train'] 
        dataloaders = dict(train=train_loader)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_n = 0
                # Iterate over data.
                pbar = tqdm(dataloaders[phase])
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * len(inputs)
                    running_corrects += torch.sum(preds == labels.data)
                    running_n += len(labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / running_n
                epoch_acc = running_corrects.double() / running_n

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {running_corrects}/{running_n} = {epoch_acc:.4f}')

                # deep copy the model
                if (phase == 'val' or not has_val) and (epoch_acc > best_acc):
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    print(f'New best accuracy: {best_acc:.3f}')
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model