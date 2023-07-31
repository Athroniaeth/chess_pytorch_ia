from torch import nn, functional


class NN(nn.Module):
    def __init__(self, input_size=768, num_classes=768):
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
