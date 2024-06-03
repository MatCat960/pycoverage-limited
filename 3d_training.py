import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture


ROBOT_RANGE = 5.0
TARGETS_NUM = 1
COMPONENTS_NUM = 1
PARTICLES_NUM = 500
AREA_W = 20.0
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 1

path = Path().resolve()
path = (path / 'logs/rnn_coverage/')

def plot_occgrid(x, y, z, save=False, name="occgrid", ax=None):
  """
  Plot heatmap of occupancy grid.
  x, y, z : meshgrid
  """
  if save:
    path = Path("/unimore_home/mcatellani/pf-training/pics/")

  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
  z_min = -1.0; z_max = 1.0
  c = ax.pcolormesh(x, y, z, cmap="YlOrRd", vmin=z_min, vmax=z_max)
  ax.set_xticks([]); ax.set_yticks([])
  if save:
    save_path = path / "{}.png".format(name)
    plt.savefig(str(save_path))
  if ax is None:
    plt.show()

def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])

  return prob



path = Path("/home/mattia/liquid_networks/datasets/3d_coverage")
files = [x for x in path.glob("**/*") if x.is_file()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

imgs = []
vels = []
FILES_NUM = len(files)//2

for i in range(FILES_NUM):
  imgs.append(np.load(str(path / f"test{i}.npy")))
  vels.append(np.load(str(path / f"vels{i}.npy")))

print("Shape of single img: ", imgs[0].shape)
print("Shape of single vels: ", vels[0].shape)
ROBOTS_NUM = imgs[0].shape[1]
GRID_STEPS = imgs[0].shape[-1]
print("Number of robots: ", ROBOTS_NUM)


#c=0

#a=torch.tensor(imgs[1])
#b=torch.tensor(imgs[2])

#c= torch.cat((a,b), dim=0)
#c.shape
'''
c=torch.empty((0,ROBOTS_NUM,64,64,64))
for i in range(len(imgs)):
  a=torch.tensor(imgs[i], dtype=torch.float32)
  #a=imgs[i]
  c=torch.cat((c,a), dim=0)

print("Shape of c: ", c.shape)


robots = []
for r in range(ROBOTS_NUM):
  robots.append(c[:, r, :, :, :])

# r1= c[:, 0, :, :]
# r2= c[:, 1, :, :]
# r3= c[:, 2, :, :]
# r4= c[:, 3, :, :]
# r5= c[:, 4, :, :]
# r6= c[:, 5, :, :]
p1 = torch.empty((0, ROBOTS_NUM, 64, 64, 64))
for r in range(ROBOTS_NUM):
  p1 = torch.cat((p1, robots[i]))
print("p1 Shape: ", p1.shape)


f=torch.empty((0, ROBOTS_NUM, 3))
for j in range(len(vels)):
  e=torch.tensor(vels[j], dtype=torch.float32)
  #a=imgs[i]
  f=torch.cat((f,e), dim=0)

vs = []
for r in range(ROBOTS_NUM):
  vs.append(f[:, r, :])

# v1= f[:, 0, :]
# v2= f[:, 1, :]
# v3= f[:, 2, :]
# v4= f[:, 3, :]
# v5= f[:, 4, :]
# v6= f[:, 5, :]

w1 = torch.empty((0, ROBOTS_NUM, 3))
for r in range(ROBOTS_NUM):
  w1 = torch.cat((w1, vs[r]))

print("w1 shape: ", w1.shape)
'''

p1 = imgs[0]
w1 = vels[0]
for i in range(1, len(imgs)):
  p1 = np.concatenate((p1, imgs[i]), 0)
  w1 = np.concatenate((w1, vels[i]), 0)
print("P1 shape: ", p1.shape)
print("w1 shape: ", w1.shape)

p1 = torch.from_numpy(p1).float()
w1 = torch.from_numpy(w1).float()

p1 = p1.view((-1, GRID_STEPS, GRID_STEPS, GRID_STEPS))
w1 = w1.view((-1, 3))

print("Final p1 shape: ", p1.shape)
print("Final vels shape: ", w1.shape)



train_size = int(p1.shape[0]*0.75)
print("Training size: ", train_size)

X_train, Y_train, X_test, Y_test = p1[:train_size], w1[:train_size], p1[train_size:], w1[train_size:]
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)
Y_train = Y_train.unsqueeze(1)
Y_test = Y_test.unsqueeze(1)
X_train, X_test, Y_train, Y_test = X_train.to(device), X_test.to(device), Y_train.to(device), Y_test.to(device)
print(f"Train/Test shapes: {X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}")

# p1.shape, w1.shape, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

from torch.utils.data import TensorDataset, DataLoader

num_classes = 3
num_epochs = 20
batch_size = 32
learning_rate = 0.001

input_size = 64
sequence_length = 64
hidden_size = 32
num_layers = 3

train_dataset = TensorDataset(X_train, Y_train)
test_dataset  = TensorDataset(X_test,  Y_test)

train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32)
test_loader  = DataLoader(test_dataset,  shuffle=False, batch_size=32)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0], batch[1]
    print(x_batch.shape, y_batch.shape)
    break


# Hyper-parameters
# input_size = 784 # 28x28


class CNN_LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(CNN_LSTM, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    # x = x.permute(0, 2, 1)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # out = out.permute(0, 2, 1)
    # print(f"shape after cnn: {out.shape}")
    out = out.view(out.shape[0], -1, out.shape[1])
    # print("Reshaped output: ", out.shape)
    out, _ = self.lstm(out)
    out = self.fc(out[:, -1, :])
    return out

class CNN_LSTM_3D(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(CNN_LSTM_3D, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2)
    )
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    # x = x.permute(0, 2, 1)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # out = out.permute(0, 2, 1)
    # print(f"shape after cnn: {out.shape}")
    out = out.view(out.shape[0], -1, out.shape[1])
    # print("Reshaped output: ", out.shape)
    out, _ = self.lstm(out)
    out = self.fc(out[:, -1, :])
    return out

cnn_lstm = CNN_LSTM_3D(input_size, hidden_size, num_layers, num_classes).to(device)



from torch import optim
# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
criterion=nn.MSELoss()

optimizer = torch.optim.Adam(cnn_lstm.parameters(), lr=learning_rate)


cnn_lstm.train()

# Train the model
total_step = len(train_loader)
loss_values=[]
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    # clear gradients for this training step
    optimizer.zero_grad()

    output = cnn_lstm(images)
    #labels=labels.squeeze(1)
    loss = criterion(output.unsqueeze(1), labels)

    # backpropagation, compute gradients
    loss.backward()
    # apply gradients
    optimizer.step()

    running_loss =+ loss.item()

        
  print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
  #if (i+1) % 619 == 0:
    # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    # print(f"Predicted: {output[epoch+1]}")
    #print(f"Lables: {labels[epoch+1]}")


  loss_values.append(running_loss)

PATH = path/'3d_model.pt'
torch.save(cnn_lstm.state_dict(), str(PATH))
print("Model saved")
