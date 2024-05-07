
# pycoverage-limited

Coverage Control of a Multi-Robot System with limited sensing range.

## Dependencies
- numpy
- [Shapely](https://shapely.readthedocs.io/en/stable/manual.html)
- [SciPy](https://scipy.org/)

## Usage
### Limited Voronoi partitioning
Calculate the bounded Voronoi partitioning of the environment and limit each Voronoi cell to the sensing range of the robot.
Bounded Voronoi partitioning of the environment:

<img src="./pics/voronoi_lim.png" alt="Voroni Limited partitioning"
    width="400"
    height="auto" />

Limited Voronoi cell for the single robot:

<img src="./pics/voronoi3.png" alt="Range-Limited Voronoi cell"
    width="400"
    height="auto" />

### Range-Limited Coverage Control
Exploit the limited Voronoi partitioning for centralized control of a Multi-Robot system with limited sensing range.

- Unifrom probability distribution
<img src="./pics/coverage_img2.png" alt="Coverage Control">

- Gaussian probability distribution
<img src="./pics/coverage_img4.png" alt="Coverage Control">
