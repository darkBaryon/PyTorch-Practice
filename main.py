import data
import torch
import model
import plot
import numpy as np
import h5py


def main(learning_rate = 0.01, batch_size = 64, epoch = 5):
    
    ## load datasets
    train_dataloader, test_dataloader = data.getData(batch_size)
    
    ## choose device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device, '\n')
    
    ## build model
    model_ = model.NeuralNetwork()
    model_.to(device)
    
    ## loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_.parameters(), lr=learning_rate)
    
    
    ## parameters
    epoch = 10
    accuracies = []
    loss = []
    
    ## training loop        
    for t in range(epoch):
        print(f"Epoch {t+1}, learning rate = {learning_rate}, batch size = {batch_size}\n-------------------------------")
        model_.trainBatch(device, train_dataloader, loss_fn, optimizer)
        loss_, accuracies_ = model_.testBatch(device, test_dataloader, loss_fn)
        
        accuracies.append(accuracies_)
        loss.append(loss_)
    print("Done!")
    
    ## save the data with h5py file
    with h5py.File(f'./data/output/h5/data_{learning_rate}_{batch_size}_{epoch}.h5', 'w') as f:

        group = f.create_group('accuracy_loss')
        
        group.create_dataset('accuracies', data=accuracies)
        group.create_dataset('losses', data=loss)
        group.create_dataset('learning_rate', data=[learning_rate])
    
    ## plot
    plot.plot_accuracy_loss(accuracies, loss, learning_rate = learning_rate, batch_size = batch_size)
    
    ## save model
    torch.save(model_.state_dict(), 'model.pth')
    print("Saved PyTorch Model State to model.pth")     



if __name__ == '__main__':
    
    for learning_rate in [0.01, 0.05, 0.1]:
        for batch_size in [64, 128, 256]:
                main(learning_rate, batch_size)