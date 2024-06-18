import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
# from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
import math

import pyvoro

from copy import deepcopy as dc
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull, Delaunay


ROBOTS_NUM = 12
ROBOT_RANGE = 5.0
TARGETS_NUM = 2
COMPONENTS_NUM = 2
PARTICLES_NUM = 500
AREA_W = 20.0
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 1


def plot_occgrid(x, y, z, save=False, name="occgrid", ax=None):
  """
  Plot heatmap of occupancy grid.
  x, y, z : meshgrid
  """
  if save:
    path = Path("/unimore_home/mcatellani/pf-training/pics/")

  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
  z_min = 0.0; z_max = 1.0
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

def gauss3d_pdf(x, y, z, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
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


def gmm3d_pdf(x, y, z, means, covs, weights):
  n_comps = len(means)
  prob = 0.0
  for i in range(n_comps):
    prob += weights[i] * gauss3d_pdf(x, y, z, means[i], covs[i])
  
  return prob

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("/unimore_home/mcatellani/pycoverage-limited/dataset2")

import torch
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Hyper-parameters
# input_size = 784 # 28x28
num_classes = 3
num_epochs = 20
#batch_size = 32
learning_rate = 0.001

input_size = 64
sequence_length = 64
hidden_size = 32
num_layers = 3

class CNN_LSTM_3D(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(CNN_LSTM_3D, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
    )
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    # self.fc = nn.Linear(hidden_size, num_classes)
    # self.fc1 = nn.Linear(64*16*16*16, 64)
    # self.fc_out = nn.Linear(64, num_classes)
    self.fc1 = nn.Linear(8192, 256*8)
    # self.fc1 = nn.Linear(64*8*8*8, 256*8)
    # self.fc2 = nn.Linear(256*8, 128)
    self.fc3 = nn.Linear(256*8, num_classes)
    self.relu = nn.ReLU()
    self.tanh = nn.Tanh()

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # print(f"shape after cnn: {out.shape}")
    # out = out.view(out.shape[0], -1, out.shape[1])
    out = out.view(out.shape[0], -1)
    # print("Reshaped output: ", out.shape)
    # out, _ = self.lstm(out)
    # out = self.fc(out[:, -1, :])
    out = self.relu(self.fc1(out))
    # out = self.relu(self.fc2(out))
    out = self.fc3(out)
    
    return out
    

class CNN_LSTM_3D_new(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(CNN_LSTM_3D_new, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
    )
    self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    # self.fc = nn.Linear(hidden_size, num_classes)
    # self.fc1 = nn.Linear(64*16*16*16, 64)
    # self.fc_out = nn.Linear(64, num_classes)
    self.fc1 = nn.Linear(128*4*4*4, 256)
    self.fc_out = nn.Linear(256, num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    #cnn takes input of shape (batch_size, channels, seq_len)
    out = self.cnn(x)
    # lstm takes input of shape (batch_size, seq_len, input_size)
    # print(f"shape after cnn: {out.shape}")
    # out = out.view(out.shape[0], -1, out.shape[1])
    out = out.view(out.shape[0], -1)
    # print("Reshaped output: ", out.shape)
    # out, _ = self.lstm(out)
    # out = self.fc(out[:, -1, :])
    out = self.relu(self.fc1(out))
    out = self.fc_out(out)

    return out

model = CNN_LSTM_3D_new(input_size, hidden_size, num_layers, num_classes).to(device)


PATH = path/'3d_model_50.pt'
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()
print(model)


for episode in range(EPISODES):
  targets = np.zeros((TARGETS_NUM, 1, 3))
  # for i in range(TARGETS_NUM):
  #   targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
  #   targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
  #   targets[i, 0, 2] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
  targets[0, 0, :] = np.array([2.5, 2.5, 2.5])
  targets[1, 0, :] = np.array([-2.5, -2.5, -2.5])

  # plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
  # plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
  # plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
  # # plt.legend()
  # plt.show()

  STD_DEV = 2.0
  samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 3))
  for k in range(TARGETS_NUM):
    for i in range(PARTICLES_NUM):
      samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 3)

  # Fit GMM
  samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 3))
  print(samples.shape)
  gmm = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
  gmm.fit(samples)

  means = gmm.means_
  covariances = gmm.covariances_
  mix = gmm.weights_

  print(f"Means: {means}")
  print(f"Covs: {covariances}")
  print(f"Mix: {mix}")


  ## -------- Generate decentralized probability grid ---------
  GRID_STEPS = 32
  s = AREA_W/GRID_STEPS     # step

  xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  zg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  Xg, Yg, Zg = np.meshgrid(xg, yg, zg)
  Xg.shape
  print(Xg.shape)

  Z = gmm3d_pdf(Xg, Yg, Zg, means, covariances, mix)
  Z = Z.reshape(GRID_STEPS, GRID_STEPS, GRID_STEPS)
  Zmax = np.max(Z)
  Z = Z / Zmax

  A_max = ROBOTS_NUM * 4*np.pi * ROBOT_RANGE**2
  phi_tot = 0.0
  dV = s**3
  for xi in xg:
    for yi in yg:
      for zi in zg:
        phi_tot += dV * gmm3d_pdf(xi, yi, zi, means, covariances, mix)

  # fig, [ax, ax2] = plt.subplots(1, 2, figsize=(12,6))
  # plot_occgrid(Xg, Yg, Z, ax=ax)
  # plot_occgrid(Xg, Yg, Z, ax=ax2)



  # ---------- Simulate episode ---------
  # ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
  ROBOTS_NUM = 12
  converged = False
  NUM_STEPS = 25
  points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 3)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []
  DISCRETIZE_PRECISION = 0.5

  imgs = np.zeros((1, ROBOTS_NUM, GRID_STEPS, GRID_STEPS, GRID_STEPS))
  vels = np.zeros((1, ROBOTS_NUM, 3))

  r_step = 2 * ROBOT_RANGE / GRID_STEPS
  opt_values = []
  for s in range(1, NUM_STEPS+1):
    print(f"*** Step {s} ***")

    conv = True
    lim_regions = []
    img_s = np.zeros((ROBOTS_NUM, GRID_STEPS, GRID_STEPS, GRID_STEPS))
    vel_s = np.zeros((ROBOTS_NUM, 3))
    opt = 0.0
    voronoi = pyvoro.compute_voronoi(points, [[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W]],2)
    phi_dq_in = 0.0
    dq_in = 0.0
    for idx in range(ROBOTS_NUM):
      # Save grid
      p_i = points[idx, :]
      xg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      yg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      zg_i = np.linspace(-ROBOT_RANGE, ROBOT_RANGE, GRID_STEPS)
      Xg_i, Yg_i, Zg_i = np.meshgrid(xg_i, yg_i, zg_i)
      Z_i = gmm3d_pdf(Xg_i, Yg_i, Zg_i, means-p_i, covariances, mix)
      Z_i = Z_i.reshape(GRID_STEPS, GRID_STEPS, GRID_STEPS)
      Zmax_i = np.max(Z_i)
      Z_i = Z_i / Zmax_i

      neighs = np.delete(points, idx, 0)
      local_pts = neighs - p_i

      # Remove undetected neighbors
      undetected = []
      for i in range(local_pts.shape[0]):
        dist = np.linalg.norm(local_pts[i])
        if dist > ROBOT_RANGE:
          undetected.append(i)

      local_pts = np.delete(local_pts, undetected, 0)

      img_i = dc(Z_i)
      for i in range(GRID_STEPS):
        for j in range(GRID_STEPS):
          for k in range(GRID_STEPS):
            # jj = GRID_STEPS-1-j
            p_ij = np.array([-ROBOT_RANGE+j*r_step, -ROBOT_RANGE+i*r_step, -ROBOT_RANGE+k*r_step])
            # print(f"Point ({i},{j}): {p_ij}")
            for n in local_pts:
              if np.linalg.norm(n - p_ij) <= SAFETY_DIST:
                img_i[i, j, k] = -1.0

            # Check if outside boundaries
            p_w = p_ij + p_i
            if p_w[0] < -0.5*AREA_W or p_w[0] > 0.5*AREA_W or p_w[1] < -0.5*AREA_W or p_w[1] > 0.5*AREA_W or p_w[2] > 0.5*AREA_W or p_w[2] < -0.5*AREA_W:
              img_i[i, j, k] = -1.0

      '''
      if idx == 0:
        print(f"Robot {idx} sees {len(local_pts)} neighbours.")
        plot_img = img_i[::2, ::2, ::2]
        xg_i = np.linspace(p_i[0]-ROBOT_RANGE, p_i[0]+ROBOT_RANGE, 16)
        yg_i = np.linspace(p_i[1]-ROBOT_RANGE, p_i[1]+ROBOT_RANGE, 16)
        zg_i = np.linspace(p_i[2]-ROBOT_RANGE, p_i[2]+ROBOT_RANGE, 16)
        xg, yg, zg = np.meshgrid(xg_i, yg_i, zg_i)
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xg, yg, zg, c=plot_img, cmap="YlOrRd", vmin=-1.0, vmax=1.0)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='v')
        ax.scatter(targets[:, 0, 0], targets[:, 0, 1], targets[:, 0, 2], marker='x', c='tab:green')
        plt.show()
      '''

      cell = voronoi[idx]
      p_i = cell['original']
      # VORONOI
      vertices = np.array(cell['vertices'])

      # get min and max for each axis
      x_min = np.min(vertices[:,0])
      x_max = np.max(vertices[:,0])
      y_min = np.min(vertices[:,1])
      y_max = np.max(vertices[:,1])
      z_min = np.min(vertices[:,2])
      z_max = np.max(vertices[:,2])

      dx = (x_max-x_min)*DISCRETIZE_PRECISION
      dy = (y_max-y_min)*DISCRETIZE_PRECISION
      dz = (z_max-z_min)*DISCRETIZE_PRECISION


      # Calculate centroid of the 3D voronoi cell
      area = 0.0
      Cx = 0.0; Cy = 0.0; Cz = 0.0
      dV = dx*dy*dz
      

      for i in np.arange(x_min, x_max, dx):
          for j in np.arange(y_min, y_max, dy):
              for k in np.arange(z_min, z_max, dz):
                  if Delaunay(vertices).find_simplex(np.array([i,j,k])) >= 0:
                      dV_pdf = dV * gmm3d_pdf(i, j, k, means, covariances, mix)
                      area += dV_pdf
                      dist = np.linalg.norm(np.array([i,j,k]) - p_i)
                      # dist_array = np.array([i,j,k] - p_i)
                      opt += dist**2 * dV_pdf
                      if dist < ROBOT_RANGE:
                        phi_dq_in += dV_pdf
                        dq_in += dV


      

      img_in = torch.from_numpy(img_i).unsqueeze(0).unsqueeze(0)
      img_in = img_in.to(torch.float).to(device)
      vel_i = model(img_in)
      # print(f"Velocity of robot {idx}: {vel_i}")
      # print("points[idx] shape: ", points[idx, :].shape)
      points[idx, 0] = points[idx, 0] + vel_i[0, 0]
      points[idx, 1] = points[idx, 1] + vel_i[0, 1]
      points[idx, 2] = points[idx, 2] + vel_i[0, 2]
    
    robots_hist = np.concatenate((robots_hist, np.expand_dims(points, 0)))
    opt_values.append(opt)
    efficiency = dq_in / A_max
    effectiveness = phi_dq_in / phi_tot
    print(f"Step: {s} | Optimization function: {opt} | Eps: {efficiency} | A: {effectiveness}")


"""
for idx in range(ROBOTS_NUM):
  cell = voronoi[idx]
  p_i = cell['original']
  

  # VORONOI
  vertices = np.array(cell['vertices'])

  # get min and max for each axis
  x_min = np.min(vertices[:,0])
  x_max = np.max(vertices[:,0])
  y_min = np.min(vertices[:,1])
  y_max = np.max(vertices[:,1])
  z_min = np.min(vertices[:,2])
  z_max = np.max(vertices[:,2])

  dx = (x_max-x_min)*DISCRETIZE_PRECISION
  dy = (y_max-y_min)*DISCRETIZE_PRECISION
  dz = (z_max-z_min)*DISCRETIZE_PRECISION


  # Calculate centroid of the 3D voronoi cell
  area = 0.0
  Cx = 0.0; Cy = 0.0; Cz = 0.0
  dV = dx*dy*dz

  for i in np.arange(x_min, x_max, dx):
      for j in np.arange(y_min, y_max, dy):
          for k in np.arange(z_min, z_max, dz):
              if Delaunay(vertices).find_simplex(np.array([i,j,k])) >= 0:
                  dV_pdf = dV * gmm3d_pdf(i, j, k, means, covariances, mix)
                  area += dV_pdf
                  dist = np.linalg.norm(np.array([i,j,k]) - p_i) ** 2
                  opt += dist * dV_pdf
                  Cx += i * dV_pdf
                  Cy += j * dV_pdf
                  Cz += k * dV_pdf

  if area == 0.0:
    break
  
  Cx = Cx / area
  Cy = Cy / area
  Cz = Cz / area



  centr = np.array([Cx, Cy, Cz]).transpose()
  robot = cell['original']
  dist = np.linalg.norm(robot-centr)
  print(f"Distance to centroid for robot {idx}: {dist}")

print(f"Optimization function: {opt}")
"""


"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(ROBOTS_NUM):
  ax.plot(robots_hist[:, i, 0], robots_hist[:, i, 1], robots_hist[:, i, 2])
  ax.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1], robots_hist[-1, i, 2])

for i in range(TARGETS_NUM):
  ax.scatter(targets[i, 0, 0], targets[i, 0, 1], targets[i, 0, 2], marker="x")

plt.xlim([-0.5*AREA_W, 0.5*AREA_W])
plt.ylim([-0.5*AREA_W, 0.5*AREA_W])
# plt.zlim([-0.5*AREA_W, 0.5*AREA_W])
plt.show()

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
for i in range(ROBOTS_NUM):
  ax2.scatter(robots_hist[-1, i, 0], robots_hist[-1, i, 1], robots_hist[-1, i, 2])

for i in range(TARGETS_NUM):
  ax2.scatter(targets[i, 0, 0], targets[i, 0, 1], targets[i, 0, 2], marker="x")

plt.show()
"""