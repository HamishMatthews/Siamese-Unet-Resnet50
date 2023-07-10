import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.cuda.amp import autocast, GradScaler

def visualize(pred, target, loss):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(pred[0, 0, :, :], cmap='gray')
    axs[0].set_title('Prediction')

    axs[1].imshow(target[0, 0, :, :], cmap='gray')
    axs[1].set_title(f'Ground Truth - Loss: {loss}')
    
    plt.show()
    
def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, num_epochs=25, accumulation_steps=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()  # For mixed precision training

    # Send the model to GPU/CPU
    if torch.cuda.device_count() > 1:
        print('Using multiple GPUs.')
        model = nn.DataParallel(model)
        criterion = nn.DataParallel(criterion)

    model = model.to(device)
    criterion = criterion.to(device)

    # To store losses for plotting
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # reset gradients
        for i, (pre_imgs, post_imgs, labels) in enumerate(train_dataloader):
            print(i)
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            labels = labels.to(device)
            
            # Forward
            with autocast():  # automatic mixed precision
                outputs = model(pre_imgs, post_imgs)
                loss = criterion(outputs, labels)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()

            # Visualization
            #visualize(outpu.detach().cpu(), target.detach().cpu(), loss)

            plt.show()
            # Check if loss is NaN and raise an error
            if torch.isnan(loss):
                print(outputs, labels)
                raise ValueError("Loss is NaN.")

            # Backward
            scaler.scale(loss).backward()

            scaler.step(optimizer)  # Make the actual update
            scaler.update()
            optimizer.zero_grad()  # Clear the gradients

            # Step the scheduler
            scheduler.step()

            running_loss += loss.item() * pre_imgs.size(0)
            # Free up some memory
            del pre_imgs
            del post_imgs
            del labels
            torch.cuda.empty_cache()

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)
        print('Train Loss: {:.4f}'.format(epoch_train_loss))

        if valid_dataloader is not None:
            # Validation phase
            model.eval()
            running_loss = 0.0
            with torch.no_grad():
                for pre_imgs, post_imgs, labels in valid_dataloader:
                    pre_imgs = pre_imgs.to(device)
                    post_imgs = post_imgs.to(device)
                    labels = labels.to(device)
                    with autocast():  # automatic mixed precision
                        outputs = model(pre_imgs, post_imgs)
                        loss = criterion(outputs, labels)
                        if torch.cuda.device_count() > 1:
                            loss = loss.mean()
                    running_loss += loss.item() * pre_imgs.size(0)
                    # Free up some memory
                    del pre_imgs
                    del post_imgs
                    del labels
                    torch.cuda.empty_cache()

            epoch_valid_loss = running_loss / len(valid_dataloader.dataset)
            valid_losses.append(epoch_valid_loss)
            print('Validation Loss: {:.4f}'.format(epoch_valid_loss))

        # Plot losses
        clear_output(wait=True)
        plt.plot(train_losses, label='Training loss')
        if valid_dataloader is not None:
            plt.plot(valid_losses, label='Validation loss')
        plt.legend()
        plt.show()

    return model

