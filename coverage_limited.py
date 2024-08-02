import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import random
from shapely import Polygon, Point, intersection

from pathlib import Path

path = Path().resolve()
img1_path = path / 'pics/coverage_img1.png'
img2_path = path / 'pics/coverage_img2.png'
img3_path = path / 'pics/coverage_img3.png'
img4_path = path / 'pics/coverage_img4.png'
img5_path = path / 'pics/coverage_img5.png'
dataset_path = path / "dataset_uniform"


ROBOTS_NUM = 12
AREA_W = 20.0
ROBOT_RANGE = 20.0
discretization_step = 0.5           # discretization of the robot's sensing range
NUM_STEPS = 100                     # number of iterations  
COLS = 3
CONVERGENCE_TOLERANCE = 0.1
NUM_EPISODES = 100

# GAUSSIAN DISTRIBUTION PARAMETERS
GAUSSIAN_DISTRIBUTION = False
GAUSS_PT = np.zeros((1, 2))         # mean point of the Gaussian distribution
GAUSS_COV = 2.0*np.eye(2)           # covariance matrix
discretize_precision = 0.5          # area discretization precision (integral calculation)
vmax = 1.5
Kp = 0.8                            # proportional gain


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



# ------------------------ UNIFORM DISTRIBUTION ------------------------
if not GAUSSIAN_DISTRIBUTION:
  for episode in range(NUM_EPISODES):
    with open(str(dataset_path/f"test{episode}.txt"), 'w') as f:
      f.write("")
    points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
    robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
    robots_hist[0, :, :] = points
    vis_regions = []
    for s in range(1, NUM_STEPS+1):
      row = 0
      if s > 2:
        row = 1
      # mirror points across each edge of the env
      dummy_points = np.zeros((5*ROBOTS_NUM, 2))
      dummy_points[:ROBOTS_NUM, :] = points
      mirrored_points = mirror(points)
      mir_pts = np.array(mirrored_points)
      dummy_points[ROBOTS_NUM:, :] = mir_pts

      # Voronoi partitioning
      vor = Voronoi(dummy_points)

      conv = True
      lim_regions = []
      for idx in range(ROBOTS_NUM):
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
        with open(str(dataset_path/f"test{episode}.txt"), 'a') as f:
          f.write(f"{robot[0]} {robot[1]}\n")
        # plt.scatter(robot[0], robot[1])

        # Intersect with robot range
        range_pts = []
        for th in np.arange(0.0, 2*np.pi, discretization_step):
          xi = robot[0] + ROBOT_RANGE * np.cos(th)
          yi = robot[1] + ROBOT_RANGE * np.sin(th)
          pt = Point(xi, yi)
          range_pts.append(pt)
          # plt.plot(xi, yi, c='tab:blue')

        range_poly = Polygon(range_pts)
        xc, yc = range_poly.exterior.xy

        lim_region = intersection(poly, range_poly)
        lim_regions.append(lim_region)
        centr = np.array([lim_region.centroid.x, lim_region.centroid.y])
        dist = np.linalg.norm(robot-centr)
        vel = Kp * (centr - robot)
        vel[0] = max(-vmax, min(vmax, vel[0]))
        vel[1] = max(-vmax, min(vmax, vel[1]))

        with open(str(dataset_path/f"test{episode}.txt"), 'a') as f:
          f.write(f"{vel[0]} {vel[1]}\n")
        points[idx, :] = robot + vel
        if dist > CONVERGENCE_TOLERANCE:
          conv = False

      # Save positions for visualization
      robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
      vis_regions.append(lim_regions)

      if conv:
        print(f"Converged in {s} iterations")
        break

  """
  fig = plt.figure(figsize=(12,10))
  plt.scatter(points[:, 0], points[:, 1])
  for region in lim_regions:
    x,y = region.exterior.xy
    plt.plot(x, y, c="tab:red")

  plt.savefig(str(img1_path))

  # Visualize motion sequence
  fig, axs = plt.subplots(2, COLS, figsize=(12,5))
  vis_step = int((robots_hist.shape[0])/COLS/2)+1
  print(f"Visualization step: {vis_step}")
  count = 0
  for i in range(0, robots_hist.shape[0], vis_step):
    r = robots_hist[i, :, :]
    region_i = vis_regions[i]
    row = 0
    if count > COLS-1:
      row = 1

    for j in range(ROBOTS_NUM):
      axs[row, count-5*row].scatter(r[j, 0], r[j, 1], c='tab:blue')
      xi, yi = region_i[j].exterior.xy
      axs[row, count-5*row].plot(xi, yi, c="tab:red")
    axs[row, count-5*row].set_xticks([]); axs[row, count-5*row].set_yticks([])
    axs[row, count-5*row].set_title(f"t = {i}")
    count += 1



  # Plot final configuration
  axs[-1, -1].scatter(points[:, 0], points[:, 1], c='tab:blue')
  axs[-1, -1].set_xticks([]); axs[-1, -1].set_yticks([])
  axs[-1, -1].set_title("Final config.")
  for region in lim_regions:
    x,y = region.exterior.xy
    axs[-1, -1].plot(x, y, c="tab:red")
  
  plt.savefig(img2_path)
  """




# ------------------------ GAUSSIAN DISTRIBUTION ------------------------
if GAUSSIAN_DISTRIBUTION:
  # generate random starting positions
  points = -0.5*AREA_W + AREA_W * np.random.rand(ROBOTS_NUM, 2)
  robots_hist = np.zeros((1, points.shape[0], points.shape[1]))
  robots_hist[0, :, :] = points
  vis_regions = []

  for s in range(1, NUM_STEPS+1):
    # mirror points across each edge of the env
    dummy_points = np.zeros((5*ROBOTS_NUM, 2))
    dummy_points[:ROBOTS_NUM, :] = points
    mirrored_points = mirror(points)
    mir_pts = np.array(mirrored_points)
    dummy_points[ROBOTS_NUM:, :] = mir_pts

    # Voronoi partitioning
    vor = Voronoi(dummy_points)

    conv = True
    lim_regions = []
    for idx in range(ROBOTS_NUM):
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
      range_pts = []
      for th in np.arange(0.0, 2*np.pi, discretization_step):
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
      A = 0.0
      Cx = 0.0; Cy = 0.0
      dA = discretize_precision ** 2
      # pts = [Point(xmin, ymin), Point(xmax, ymin), Point(xmax, ymax), Point(xmin, ymax)]
      # bound = Polygon(pts)
      for i in np.arange(xmin, xmax, discretize_precision):
        for j in np.arange(ymin, ymax, discretize_precision):
          pt_i = Point(i,j)
          if lim_region.contains(pt_i):
            dA_pdf = dA * gauss_pdf(i, j, GAUSS_PT, GAUSS_COV)
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
      vel = Kp * (centr - robot)
      vel[0,0] = max(-vmax, min(vmax, vel[0,0]))
      vel[0,1] = max(-vmax, min(vmax, vel[0,1]))

      points[idx, :] = robot + vel
      if dist > CONVERGENCE_TOLERANCE:
        conv = False

    # Save positions for visualization
    if s == 1:
      vis_regions.append(lim_regions)
    robots_hist = np.vstack((robots_hist, np.expand_dims(points, axis=0)))
    vis_regions.append(lim_regions)

    if conv:
      print(f"Converged in {s} iterations")
      break
    # axs[row, s-1-5*row].scatter(points[:, 0], points[:, 1])


  plt.scatter(points[:, 0], points[:, 1])
  for region in lim_regions:
    x,y = region.exterior.xy
    plt.plot(x, y, c="tab:red")

  plt.savefig(str(img3_path))


  # Visualize motion sequence
  fig, axs = plt.subplots(2, COLS, figsize=(12,5))
  vis_step = int((robots_hist.shape[0])/COLS/2)
  # print(vis_step)
  # print(0.5*robots_hist.shape[0] % COLS)
  if (0.5*robots_hist.shape[0] % COLS > 0.5):
    vis_step += 1 
  print(f"Visualization step: {vis_step}")
  # print(f"Robots hist shape : {robots_hist.shape[0]}")
  count = 0
  for i in range(0, robots_hist.shape[0], vis_step):
    r = robots_hist[i, :, :]
    region_i = vis_regions[i]
    row = 0
    if count > COLS-1:
      row = 1

    for j in range(ROBOTS_NUM):
      axs[row, count-5*row].scatter(r[j, 0], r[j, 1], c='tab:blue')
      xi, yi = region_i[j].exterior.xy
      axs[row, count-5*row].plot(xi, yi, c="tab:red")
    axs[row, count-5*row].set_xticks([]); axs[row, count-5*row].set_yticks([])
    axs[row, count-5*row].set_title(f"t = {i}")
    count += 1



  # Plot final configuration
  axs[-1, -1].scatter(points[:, 0], points[:, 1], c='tab:blue')
  axs[-1, -1].set_xticks([]); axs[-1, -1].set_yticks([])
  axs[-1, -1].set_title("Final config.")
  for region in lim_regions:
    x,y = region.exterior.xy
    axs[-1, -1].plot(x, y, c="tab:red")

  plt.savefig(str(img4_path))