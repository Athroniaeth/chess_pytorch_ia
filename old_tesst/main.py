import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.detection import transform
from torchvision.transforms import transforms


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.function_1 = nn.Linear(input_size, 256)
        self.function_2 = nn.Linear(256, 256)
        self.function_3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.function_1(x)
        x = functional.relu(x)

        x = self.function_2(x)
        x = functional.relu(x)

        x = self.function_3(x)
        x = functional.softmax(x, dim=1)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.__version__, torch.version.cuda)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print(f'Max: {round(torch.cuda.max_memory_reserved() / 1024 ** 3, 1)}')
print(device)

input_size = 784
num_classes = 10

lr = 5e-4
batch_size = 128
num_epochs = 5

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size, num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

from tqdm import tqdm

for epoch in range(num_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (data, targets) in pbar:
        data = data.to(device=device)
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0], -1)

        scores = model.forward(data)

        _, predictions = scores.max(1)
        accuracy = (predictions == targets).sum()
        accuracy = f"{(float(accuracy) / batch_size) * 100:.2f}"

        loss = criterion(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        pbar.set_description(f"epochs: {epoch + 1}/{num_epochs}  -  accuracy : {accuracy}  -  loss: {loss.item():.4f}")


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        calcul = f"{float(num_correct) / float(num_samples) * 100:.2f}"

        print(f"Got {num_correct} / {num_samples} with accuracy {calcul}")

        model.train()
        return calcul


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
