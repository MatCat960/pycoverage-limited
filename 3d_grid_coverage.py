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
TARGETS_NUM = 4
PARTICLES_NUM = 500
AREA_W = 40.0
vmax = 1.5
SAFETY_DIST = 2.0
EPISODES = 100
CONVERGENCE_TOLERANCE = 0.5
DISCRETIZE_PRECISION = 0.5
NUM_STEPS = 100
dt = 0.2

path = Path().resolve()
path = (path / 'dataset')


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



for episode in range(EPISODES):
  targets = np.zeros((TARGETS_NUM, 1, 3))
  for i in range(TARGETS_NUM):
    targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
    targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
    targets[i, 0, 2] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)

  # plt.plot([-0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, -0.5*AREA_W], c='tab:blue', label="Environment")
  # plt.plot([0.5*AREA_W, 0.5*AREA_W], [-0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, 0.5*AREA_W], c='tab:blue')
  # plt.plot([-0.5*AREA_W, -0.5*AREA_W], [0.5*AREA_W, -0.5*AREA_W], c='tab:blue')
  # plt.scatter(targets[:, :, 0], targets[:, :, 1], c='tab:orange', label="Targets")
  # # plt.legend()
  # plt.show()

  STD_DEV = 2.0 * 2*np.random.rand()
  samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 3))
  for k in range(TARGETS_NUM):
    for i in range(PARTICLES_NUM):
      samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 3)

  # Fit GMM
  samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 3))
  print(samples.shape)
  gmm = GaussianMixture(n_components=TARGETS_NUM, covariance_type='full', max_iter=1000)
  gmm.fit(samples)

  means = gmm.means_
  covariances = gmm.covariances_
  mix = gmm.weights_

  print(f"Means: {means}")
  print(f"Covs: {covariances}")
  print(f"Mix: {mix}")


  ## Generate probability grid
  GRID_STEPS = 64
  s = AREA_W/GRID_STEPS     # step

  xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  zg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
  Xg, Yg, Zg = np.meshgrid(xg, yg, zg)
  Xg.shape

  Z = gmm3d_pdf(Xg, Yg, Zg, means, covariances, mix)
  Z = Z.reshape(GRID_STEPS, GRID_STEPS, GRID_STEPS)
  Zmax = np.max(Z)
  Z = Z / Zmax
  print("Z shape: ", Z.shape)


  # Simulate episode
  # ROBOTS_NUM = np.random.randint(6, ROBOTS_MAX)
  converged = False
  points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 3)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []

  imgs = np.zeros((1, ROBOTS_NUM, GRID_STEPS, GRID_STEPS, GRID_STEPS))
  vels = np.zeros((1, ROBOTS_NUM, 3))

  r_step = 2 * ROBOT_RANGE / GRID_STEPS
  for s in range(1, NUM_STEPS+1):
    voronoi = pyvoro.compute_voronoi(points, [[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W],[-0.5*AREA_W, 0.5*AREA_W]],2)

    conv = True
    lim_regions = []
    img_s = np.zeros((ROBOTS_NUM, GRID_STEPS, GRID_STEPS, GRID_STEPS))
    vel_s = np.zeros((ROBOTS_NUM, 3))
    for idx in range(ROBOTS_NUM):
      # Save grid
      cell = voronoi[idx]
      p_i = cell['original']
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
        if local_pts[i, 0] < -ROBOT_RANGE or local_pts[i, 0] > ROBOT_RANGE or local_pts[i, 1] < -ROBOT_RANGE or local_pts[i, 1] > ROBOT_RANGE or local_pts[i, 2] < -ROBOT_RANGE or local_pts[i, 2] > ROBOT_RANGE:
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
          if p_w[0] < -0.5*AREA_W or p_w[0] > 0.5*AREA_W or p_w[1] < -0.5*AREA_W or p_w[1] > 0.5*AREA_W or p_w[2] < 0.5*AREA_W or p_w[2] > 0.5*AREA_W:
            img_i[i, j, k] = -1.0
        
      img_s[idx, :, :, :] = img_i

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
                      Cx += i * dV_pdf
                      Cy += j * dV_pdf
                      Cz += k * dV_pdf

      if area == 0.0:
        break
        
      Cx = Cx / area
      Cy = Cy / area
      Cz = Cz / area

      centr = np.array([Cx, Cy, Cz]).transpose()
      # print(f"Robot: {robot}")
      # print(f"Centroid: {centr}")
      robot = cell['original']
      dist = np.linalg.norm(robot-centr)
      if dist > CONVERGENCE_TOLERANCE:
          conv = False

      vel = 0.8 * (centr - robot)
      vel[0,0] = max(-vmax, min(vmax, vel[0,0]))
      vel[0,1] = max(-vmax, min(vmax, vel[0,1]))
      vel[0,2] = max(-vmax, min(vmax, vel[0,2]))
      vel_s[idx, :] = vel
      points[idx, :] = robot + vel*dt
      '''
      region = vor.point_region[idx]
      poly_vert = []
      for vert in vor.regions[region]:
        v = vor.vertices[vert]
        poly_vert.append(v)
        # plt.scatter(v[0], v[1], c='tab:red')

      poly = Polygon(poly_vert)
      x,y = poly.exterior.xy
      # plt.plot(x, y, c='tab:orange')
      # robot = np.array([-18.0, -12.0])
      robot = vor.points[idx]

      # plt.scatter(robot[0], robot[1])

      # Intersect with robot range
      step = 0.5
      range_pts = []
      for th in np.arange(0.0, 2*np.pi, step):
        xi = robot[0] + ROBOT_RANGE * np.cos(th)
        yi = robot[1] + ROBOT_RANGE * np.sin(th)
        pt = Point(xi, yi)
        range_pts.append(pt)
        # plt.plot(xi, yi, c='tab:blue')

      range_poly = Polygon(range_pts)
      xc, yc = range_poly.exterior.xy

      lim_region = intersection(poly, range_poly)
      lim_regions.append(lim_region)

      # Calculate centroid with gaussian distribution
      xmin, ymin, xmax, ymax = lim_region.bounds
      # print(f"x range: {xmin} - {xmax}")
      # print(f"y range: {ymin} - {ymax}")
      A = 0.0
      Cx = 0.0; Cy = 0.0
      dA = discretize_precision ** 2
      # pts = [Point(xmin, ymin), Point(xmax, ymin), Point(xmax, ymax), Point(xmin, ymax)]
      # bound = Polygon(pts)
      for i in np.arange(xmin, xmax, discretize_precision):
        for j in np.arange(ymin, ymax, discretize_precision):
          pt_i = Point(i,j)
          if lim_region.contains(pt_i):
            dA_pdf = dA * gmm_pdf(i, j, means, covariances, mix)
            # print(dA_pdf)
            A = A + dA_pdf
            Cx += i*dA_pdf
            Cy += j*dA_pdf

      Cx = Cx / A
      Cy = Cy / A



      # centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
      centr = np.array([Cx, Cy]).transpose()
      # print(f"Robot: {robot}")
      # print(f"Centroid: {centr}")
      dist = np.linalg.norm(robot-centr)
      vel = 0.8 * (centr - robot)
      vel[0, 0] = max(-vmax, min(vmax, vel[0,0]))
      vel[0, 1] = max(-vmax, min(vmax, vel[0,1]))
      vel_s[idx, :] = vel

      points[idx, :] = robot + vel
      if dist > 0.1:
        conv = False
      '''


    
    imgs = np.concatenate((imgs, np.expand_dims(img_s, 0)))
    vels = np.concatenate((vels, np.expand_dims(vel_s, 0)))

    # Save positions for visualization
    if s == 1:
      vis_regions.append(lim_regions)
    robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
    vis_regions.append(lim_regions)

    if conv:
      print(f"Converged in {s} iterations")
      break
    # axs[row, s-1-5*row].scatter(points[:, 0], points[:, 1])

  imgs = imgs[1:]
  vels = vels[1:]
  imgs.shape

  with open(str(path/f"test{episode}.npy"), 'wb') as f:
    np.save(f, imgs)
  with open(str(path/f"vels{episode}.npy"), 'wb') as f:
    np.save(f, vels)