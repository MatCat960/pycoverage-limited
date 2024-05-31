import pyvoro
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
# from shapely import Polygon, Point, intersection
from tqdm import tqdm
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, Delaunay
from sklearn.mixture import GaussianMixture

# SIMULATION PARAMS
NUM_EPOCHS = 1
ROBOTS_NUM = 12
AREA_W = 40.0
vmax = 1.5
NUM_STEPS = 1000
GAUSSIAN_DISTRIBUTION = True
DISCRETIZE_PRECISION = 0.2
CONVERGENCE_TOLERANCE = 0.5
TARGETS_NUM = 4
PARTICLES_NUM = 500
dt = 0.2

path = Path().resolve()

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

def gmm3d_pdf(x, y, z, means, covs, weights):
  n_comps = len(means)
  prob = 0.0
  for i in range(n_comps):
    prob += weights[i] * gauss3d_pdf(x, y, z, means[i], covs[i])
  
  return prob


points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 3)
# cov = np.eye(3)
# GAUSS_PT = np.random.uniform(-0.5*AREA_W, 0.5*AREA_W, 3)
STD_DEV = 2.0 + 2*np.random.rand()
targets = np.zeros((TARGETS_NUM, 1, 3))
for i in range(TARGETS_NUM):
  targets[i, 0, 0] = -0.5*AREA_W + 1.0 + (AREA_W-2) * np.random.rand()
  targets[i, 0, 1] = -0.5*AREA_W + 1.0 + (AREA_W-2) * np.random.rand()
  targets[i, 0, 2] = -0.5*AREA_W + 1.0 + (AREA_W-2) * np.random.rand()

samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 3))
for k in range(TARGETS_NUM):
  for i in range(PARTICLES_NUM):
    samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 3)
    # plt.scatter(samples[k, i, 0], samples[k, i, 1], samples[k, i, 2], s=0.2, c="tab:orange")

# Fit GMM
samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 3))
# print(samples.shape)
gmm = GaussianMixture(n_components=TARGETS_NUM, covariance_type='full', max_iter=1000)
gmm.fit(samples)

means = gmm.means_
covariances = gmm.covariances_
mix = gmm.weights_

pts_hist = np.zeros((NUM_STEPS, ROBOTS_NUM, 3))
pts_hist[0] = points
print(pts_hist.shape)


if GAUSSIAN_DISTRIBUTION:
    for s in tqdm(range(NUM_STEPS)):
        voronoi = pyvoro.compute_voronoi(points,[[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W]],2)
        # print(voronoi)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for each Voronoi cell, plot all the faces of the corresponding polygon
        v = 0
        conv = True
        for vnoicell in voronoi:
          faces = []
          # the vertices are the corner points of the Voronoi cell
          vertices = np.array(vnoicell['vertices'])
          p = vnoicell['original']

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
                          Cx += i * dV_pdf
                          Cy += j * dV_pdf
                          Cz += k * dV_pdf

          Cx = Cx / area
          Cy = Cy / area
          Cz = Cz / area

          centr = np.array([Cx, Cy, Cz]).transpose()
          # print(f"Robot: {robot}")
          # print(f"Centroid: {centr}")
          robot = vnoicell['original']
          dist = np.linalg.norm(robot-centr)
          if dist > CONVERGENCE_TOLERANCE:
              conv = False

          vel = 0.8 * (centr - robot)
          vel[0,0] = max(-vmax, min(vmax, vel[0,0]))
          vel[0,1] = max(-vmax, min(vmax, vel[0,1]))
          vel[0,2] = max(-vmax, min(vmax, vel[0,2]))
          points[v, :] = robot + vel*dt

          pts_hist[s, v, :] = robot + vel
          v += 1




        if conv:
            print(f"Converged in {s} iterations.")
            break


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# for i in range(pts_hist.shape[0]):
for j in range(ROBOTS_NUM):
  ax.plot(pts_hist[:s,j,0], pts_hist[:s,j,1], pts_hist[:s,j,2])

plt.savefig(str(path/"pics/3d_traj.png"))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for pt in points:
    ax.scatter(pt[0], pt[1], pt[2])
for i in range(TARGETS_NUM):
  ax.scatter(targets[i, 0, 0], targets[i, 0, 1], targets[i, 0, 2], c='r', marker='x')

plt.savefig(str(path/"pics/3d_final.png"))