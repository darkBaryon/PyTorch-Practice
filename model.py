import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        
        print("NeuralNetwork init")
        
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    ## functions        
    def trainBatch(self, device, train_dataloader, loss_fn, optimizer):
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = self(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def testBatch(self, device, test_dataloader, loss_fn):
        size = len(test_dataloader.dataset)
        num_batches = len(test_dataloader)
        self.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        return test_loss, correct
    

if __name__ == '__main__':
    print("NeuralNetwork main")
    model = NeuralNetwork()
    print(model)
    
    model.train()