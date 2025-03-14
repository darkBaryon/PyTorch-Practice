import matplotlib.pyplot as plt
import numpy as np
import h5py

def plot_accuracy_loss(accuracies, losses, learning_rate = 0.05, batch_size = 64):
    
    epoch = np.arange(1, len(np.array(accuracies))+1)

    fig, ax1 = plt.subplots()
    ax1.plot([acc*100 for acc in accuracies], 'b-', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim(0, 100)  # 设置纵轴取值范围为0-100
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(losses, 'r-', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.xticks(epoch)
    plt.title('Test Accuracy and Loss over Epochs')
    if learning_rate != None:
        plt.text(0.95, 0.95, f'learning rate = {learning_rate}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.legend(loc='upper left')
    
    plt.savefig(f'./data/output/figure/accuracy_loss_{learning_rate}_{batch_size}_{epoch[-1]}.png')
    print(f'Saved accuracy and loss to ./data/output/figure/accuracy_loss_{learning_rate}_{batch_size}_{epoch[-1]}.png')
    
    
    
    
if __name__ == "__main__":
    
    with h5py.File('./data/output/data_0.05.h5', 'r') as f:
        print(f['accuracy_loss'].keys())
        group = f['accuracy_loss']
        accuracies = group['accuracies'][:]
        losses = group['losses'][:]
        learning_rate = group['learning_rate'][0]
    
    print(learning_rate)
    print(accuracies)
    print(losses)
    
    plot_accuracy_loss(accuracies, losses, learning_rate = learning_rate)