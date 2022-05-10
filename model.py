import numpy as np
import os
import torch
import torch.nn.functional as F
from accuracies import *
from generate_plot import *


class FER_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=32, out_channels=7, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=4)
        self.norm1 = torch.nn.BatchNorm2d(32)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.norm4 = torch.nn.BatchNorm2d(7)
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 48, 48)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm1(self.conv2(x)))
        x = F.relu(self.norm1(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.norm2(self.conv3(x)))
        x = F.relu(self.norm2(self.conv4(x)))
        x = F.relu(self.norm2(self.conv4(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.norm3(self.conv5(x)))
        x = F.relu(self.norm3(self.conv6(x)))
        x = F.relu(self.norm3(self.conv6(x)))
        x = self.pool1(x)
        x = F.relu(self.norm1(self.conv7(x)))
        x = F.relu(self.norm4(self.conv8(x)))
        x = self.pool2(x)
        x = x.reshape(x.shape[0], 7)
        return x


if __name__ == "__main__":
    DATA_DIR = "datasets"
    IMG_TRAIN_FILE = "train_images.npy"        
    IMG_DEV_FILE = "dev_images.npy"
    IMG_TEST_FILE = "test_images.npy"
    LABEL_TRAIN_FILE = "train_labels.npy"
    LABEL_DEV_FILE = "dev_labels.npy"
    LABEL_TEST_FILE = "test_labels.npy"

    MODEL_SAVE_FILE = "best_model"
    GRAPH_SAVE_FILE = "best_graph"
    LEARNING_RATE = .05
    BATCH_SIZE = 256
    EPOCHS = 1000

    train_acc_arr = np.full(int(EPOCHS/100), 0, dtype=float)
    dev_acc_arr = np.full(int(EPOCHS/100), 0, dtype=float)
    test_acc_arr = np.full(int(EPOCHS/100), 0, dtype=float)

    flat_train_images = np.load(os.path.join(DATA_DIR, IMG_TRAIN_FILE))
    flat_dev_images = np.load(os.path.join(DATA_DIR, IMG_DEV_FILE))
    flat_test_images = np.load(os.path.join(DATA_DIR, IMG_TEST_FILE))
    train_labels = np.load(os.path.join(DATA_DIR, LABEL_TRAIN_FILE), allow_pickle=True)
    dev_labels = np.load(os.path.join(DATA_DIR, LABEL_DEV_FILE), allow_pickle=True)
    test_labels = np.load(os.path.join(DATA_DIR, LABEL_TEST_FILE), allow_pickle=True)
    
    N_IMAGES = flat_train_images.shape[0]
    DIM = 48

    model = FER_CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for step in range(EPOCHS):
        i = np.random.choice(flat_train_images.shape[0], size=BATCH_SIZE, replace=False)
        x = torch.from_numpy(flat_train_images[i].astype(np.float32))
        y = torch.from_numpy(train_labels[i].astype(int))
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_images, train_labels)
            dev_acc, dev_loss = dev_acc_and_loss(model, flat_dev_images, dev_labels)
            test_acc, test_loss = dev_acc_and_loss(model, flat_test_images, test_labels)

            train_acc_arr[int(step/100)] = train_acc
            dev_acc_arr[int(step/100)] = dev_acc
            test_acc_arr[int(step/100)] = test_acc

            step_metrics = {
                'step': step, 
                'train_loss': loss.item(),                   
                'train_acc': train_acc,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc,
                'test_loss': test_loss,
                'test_acc': test_acc
            }

            print(f"On step {step}:\tTrain loss {train_loss}\t| Dev acc {dev_acc}\t")
    generate_plot(train_acc_arr, dev_acc_arr, test_acc_arr, GRAPH_SAVE_FILE)
    torch.save(model, MODEL_SAVE_FILE)