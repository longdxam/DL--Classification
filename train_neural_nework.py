from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms


import torch
import torch.nn as nn
class simpleNN(nn.Module):
    def __init__(self, num_class = 10): #dinh nghia cac layer can dung
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=3*32*32, out_features= 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features= 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features= 1024),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features= 512),
            nn.ReLU()
        )
        self.fc5= nn.Sequential(
            nn.Linear(in_features=512, out_features = num_class),
            nn.ReLU()
        )

    def forward(self, x): #cach thuc du lieu di qua nhu nao
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class CNN(nn.Module):
 def __init__(self, num_class=10):  # dinh nghia cac layer can dung
  super().__init__()
  self.conv1 = self.make_block(in_channels=3, out_channels=8)
  self.conv2 = self.make_block(in_channels=8, out_channels=16)
  self.conv3 = self.make_block(in_channels=16, out_channels=32)
  self.conv4 = self.make_block(in_channels=32, out_channels=64)
  self.conv5 = self.make_block(in_channels=64, out_channels=128)
  # self.flatten = nn.Flatten()

  self.fc1 = nn.Sequential(
   nn.Dropout(p=0.5),
   nn.Linear(in_features=6272, out_features=512),
   nn.LeakyReLU()
  )

  self.fc2 = nn.Sequential(
   nn.Dropout(p=0.5),
   nn.Linear(in_features=512, out_features=1024),
   nn.LeakyReLU()
  )

  self.fc3 = nn.Sequential(
   nn.Dropout(p=0.5),
   nn.Linear(in_features=1024, out_features=num_class),
  )

 def make_block(self, in_channels, out_channels):
  return nn.Sequential(
   nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
   nn.BatchNorm2d(num_features=out_channels),
   nn.LeakyReLU(),
   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same'),
   nn.BatchNorm2d(num_features=out_channels),
   nn.LeakyReLU(),
   nn.MaxPool2d(kernel_size=2)
  )

 def forward(self, x):  # cach thuc du lieu di qua nhu nao
  x = self.conv1(x)
  x = self.conv2(x)
  x = self.conv3(x)
  x = self.conv4(x)
  x = self.conv5(x)
  x = x.view(x.shape[0], -1)  # flatten
  x = self.fc1(x)
  x = self.fc2(x)
  x = self.fc3(x)
  return x

transform = transforms.Compose([
    transforms.ToTensor(),  # chuyển ảnh từ PIL sang tensor
])

train_dataset = CIFAR10(root = './', train = True, transform = transform)
train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = 16,
    num_workers = 3,
    shuffle = True,
    drop_last = True,
)

test_dataset = CIFAR10(root = './', train = False,transform = transform)
test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = 16,
    num_workers = 3,
    shuffle = False,
    drop_last = True,
)

model = CNN()
input_data = torch.rand(8,3,224,224)
result = model(input_data)
result.shape

criterion = nn.CrossEntropyLoss() #dinh nghia ham loss
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

num_epochs = 100
num_iters = len(train_dataloader)

for epoch in range(num_epochs):
 model.train()  # model dang trong qua trinh train
 for iter, (images, labels) in enumerate(train_dataloader):
  # forward
  outputs = model(images)
  loss_value = criterion(outputs, labels)
  print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch + 1, num_epochs, iter + 1, num_iters, loss_value))

  # backword
  optimizer.zero_grad()  # ko luu dao ham gradien
  loss_value.backward()  # tinh gradien cua ham loss
  optimizer.step()  # cap nhat tham so

 model.eval()  # test
 all_predictions = []
 op
 all_labels = []
 for iter, (images, labels) in enumerate(test_dataloader):
  all_labels.extend(labels)
  with torch.no_grad():  # tat ca lenh trong nay se ko dc tinh gradien
   predictions = model(images)
   indices = torch.argmax(predictions, dim=1)
   all_predictions.extend(indices)
   loss_value = criterion(outputs, labels)

 all_labels = [label.item() for label in all_labels]
 all_predictions = [prediction.item() for prediction in all_predictions]
 correct = sum([pred == label for pred, label in zip(all_predictions, all_labels)])
 total = len(all_labels)
 accuracy = correct / total

 print("Epoch {}: Accuracy on test set = {:.2f}%".format(epoch + 1, accuracy * 100))