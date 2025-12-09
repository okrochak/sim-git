import numpy as np
import random
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm
# ===========================
# Configuration
# ===========================
NUM_PARTICLES = 10 #1000000
OUTPUT_FILE = "part.txt"
PARTICLE_DENSITY = 1000.0  # constant across all particles
MAX_VELOCITY_MAGNITUDE = 0.15  # max speed for velocity sampling
# ===========================
# Check if point is inside STL
# ===========================
def filter_points_outside_stl(
    input_file: str,
    output_file: str,
    stl_path: str,
    batch_size: int = 50000
):
    # Load STL once
    mesh = trimesh.load(stl_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Load the particle table
    data = np.loadtxt(input_file, delimiter="\t")

    # First 3 columns are x,y,z
    coords = data[:, 2:5]

    keep_mask = np.ones(len(coords), dtype=bool)

    # Process in chunks to avoid memory spikes
    for start in tqdm(range(0, len(coords), batch_size)):
        end = start + batch_size
        chunk = coords[start:end]
        inside = mesh.contains(chunk)  # vectorized
        keep_mask[start:end] = ~inside

    filtered = data[keep_mask]
    np.savetxt(output_file, filtered, delimiter="\t", fmt="%.6f")

    print(f"Filtered {len(data) - len(filtered)} points inside mesh. "
          f"Kept {len(filtered)} points.")

# ===========================
# Diameter distribution
# ===========================
def sample_diameter(method, **kwargs):
    if method == "log-normal":
        mean = kwargs["mean"]; sigma = kwargs["sigma"]
        # log-normal distribution
        # return 1E-6*np.random.lognormal(mean=mean, sigma=sigma)
        return 1E-4*np.random.lognormal(mean=mean, sigma=sigma)
# ===========================
# Position sampling
# ===========================
def sample_position(method, **kwargs):
    if method == "box":
        x = random.uniform(kwargs["xmin"], kwargs["xmax"])
        y = random.uniform(kwargs["ymin"], kwargs["ymax"])
        z = random.uniform(kwargs["zmin"], kwargs["zmax"])
        return [x, y, z]

    elif method == "sphere":
        radius = kwargs.get("radius")
        x = kwargs["x"]; y = kwargs["y"]; z = kwargs["z"]
        while True:
            point = np.random.uniform(-radius, radius, size=3)
            if np.linalg.norm(point) <= radius:
                return (point + np.array([x, y, z])).tolist()
        
    else:
        raise ValueError(f"Unknown position method: {method}")

# ===========================
# Velocity sampling
# ===========================
def sample_velocity(max_magnitude):
    direction = np.random.normal(0, 1, 3)
    direction /= np.linalg.norm(direction)
    magnitude = random.uniform(0, max_magnitude)
    return (direction * magnitude).tolist()

# ===========================
# Main
# ===========================
def generate_particles(n_particles, output_file):
    with open(output_file, "w") as f:
        for _ in tqdm(range(n_particles)):
            diameter = sample_diameter(method="log-normal", mean=5.167, sigma=0.75)
            density = PARTICLE_DENSITY
            position = sample_position(
                # method="sphere",
                # radius=5.0,
                # x=15.0, 
                # y=-275.0,
                # z=180.0
                method="box",
                xmin=-20.0, xmax=20.0,
                ymin=-80.0, ymax=-40.0,
                zmin=-40.0, zmax=-0.0
            )
            velocity = sample_velocity(MAX_VELOCITY_MAGNITUDE)
            row = [diameter, density] + position # + velocity
            # row = [diameter, density] + position + velocity

            f.write("\t".join(f"{val:.6f}" for val in row) + "\n")

# ===========================
# Plotting Routine
# ===========================
def plot_distributions(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    diameters = data[:, 0]
    densities = data[:, 1]
    positions = data[:, 2:5]
    velocities = data[:, 5:8]

    # Density plot
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(diameters, bins=30, density=True, color="gray")
    plt.title("PDF of Diameter")
    plt.xlabel("Diameter")
    plt.ylabel("Probability Density")
    # Position components
    plt.subplot(1, 3, 2)
    for i, label in enumerate(["x", "y", "z"]):
        plt.hist(positions[:, i], bins=30, density=True, alpha=0.6, label=label)
    plt.title("PDF of Initial Position Components")
    plt.xlabel("Position")
    plt.ylabel("Probability Density")
    plt.legend()
    # Velocity components
    plt.subplot(1, 3, 3)
    for i, label in enumerate(["x", "y", "z"]):
        plt.hist(velocities[:, i], bins=30, density=True, alpha=0.6, label=label)
    plt.title("PDF of Initial Velocity Components")
    plt.xlabel("Velocity")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig("velocity_distribution.png")
    # plt.show()

if __name__ == "__main__":
    generate_particles(NUM_PARTICLES, OUTPUT_FILE)
    # filter_points_outside_stl(OUTPUT_FILE, "filtered_" + OUTPUT_FILE, "stl/nose.stl")
    # plot_distributions(OUTPUT_FILE)
