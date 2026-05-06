import numpy as np
import random
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm

# ===========================
# Velocity sampling
# ===========================
def sample_velocity(max_magnitude):
    direction = np.random.normal(0, 1, 3)
    direction /= np.linalg.norm(direction)
    magnitude = random.uniform(0, max_magnitude)
    return (direction * magnitude).tolist()


# ===========================
# Particles class
# ===========================
class Particles:
    def __init__(self, n_particles, output_file):
        self.n_particles = n_particles
        self.output_file = output_file
        self.positions = None
        self.diameters = None

    def sample_position(self, method, **kwargs):
        if method == "box":
            x = np.random.uniform(kwargs["xmin"], kwargs["xmax"], self.n_particles)
            y = np.random.uniform(kwargs["ymin"], kwargs["ymax"], self.n_particles)
            z = np.random.uniform(kwargs["zmin"], kwargs["zmax"], self.n_particles)
            self.positions = np.column_stack((x, y, z))

        elif method == "stl":
            stl_file = kwargs["stl_file"]
            normal_offset = kwargs.get("normal_offset", 3.0e-4)
            tangential_jitter = kwargs.get("tangential_jitter", 1.0e-5)
            flip_normal = kwargs.get("flip_normal", False)

            mesh = trimesh.load_mesh(stl_file, process=False)
            if mesh.is_empty:
                raise RuntimeError(f"Loaded mesh is empty: {stl_file}")

            points, face_idx = trimesh.sample.sample_surface(mesh, self.n_particles)
            face_normals = mesh.face_normals[face_idx]

            scale = kwargs.get("scale", 1.0)
            if scale != 1.0:
                center = mesh.vertices.mean(axis=0)
                points = center + scale * (points - center)

            sign = 1.0 if flip_normal else -1.0
            points = points + sign * normal_offset * face_normals
            if tangential_jitter > 0.0:
                points = points + np.random.normal(0.0, tangential_jitter, size=points.shape)
            self.positions = points

        else:
            raise ValueError(f"Unknown position method: {method}")

    def sample_diameter(self, method, **kwargs):
        if method == "log-normal":
            mean = kwargs.get("mean", 0.0)
            sigma = kwargs.get("sigma", 1.0)
            self.diameters = 1e-4 * np.random.lognormal(mean=mean, sigma=sigma, size=self.n_particles)

        elif method == "uniform":
            choices = kwargs.get("choices")
            if choices is None:
                raise ValueError("The 'uniform' method requires a 'choices' argument.")
            self.diameters = np.random.choice(choices, size=self.n_particles)

        else:
            raise ValueError(f"Unknown diameter method: {method}")

    def generate(self):
        if self.positions is None:
            raise RuntimeError("Call sample_position() before generate().")
        if self.diameters is None:
            raise RuntimeError("Call sample_diameter() before generate().")

        with open(self.output_file, "w") as f:
            for i in tqdm(range(self.n_particles)):
                row = [self.diameters[i], PARTICLE_DENSITY] + self.positions[i].tolist()
                # row += sample_velocity(MAX_VELOCITY_MAGNITUDE)
                f.write("\t".join(f"{val:.6f}" for val in row) + "\n")

    def save_vtk(self, vtk_file=None):
        if self.positions is None or self.diameters is None:
            raise RuntimeError("Call sample_position() and sample_diameter() before save_vtk().")

        if vtk_file is None:
            vtk_file = self.output_file.rsplit(".", 1)[0] + ".vtk"

        n = self.n_particles
        densities = np.full(n, PARTICLE_DENSITY)

        with open(vtk_file, "w") as f:
            f.write("# vtk DataFile Version 2.0\n")
            f.write("Particles\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {n} float\n")
            for p in self.positions:
                f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            f.write(f"VERTICES {n} {2 * n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {n}\n")
            f.write("SCALARS diameter float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for d in self.diameters:
                f.write(f"{d:.6e}\n")
            f.write("SCALARS density float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for rho in densities:
                f.write(f"{rho:.6e}\n")

        print(f"Wrote VTK file: {vtk_file}")


# ===========================
# Plotting Routine
# ===========================
def plot_distributions(file_path):
    data = np.loadtxt(file_path, delimiter="\t")
    diameters = data[:, 0]
    positions = data[:, 2:5]
    velocities = data[:, 5:8]

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(diameters, bins=30, density=True, color="gray")
    plt.title("PDF of Diameter")
    plt.xlabel("Diameter")
    plt.ylabel("Probability Density")

    plt.subplot(1, 3, 2)
    for i, label in enumerate(["x", "y", "z"]):
        plt.hist(positions[:, i], bins=30, density=True, alpha=0.6, label=label)
    plt.title("PDF of Initial Position Components")
    plt.xlabel("Position")
    plt.ylabel("Probability Density")
    plt.legend()

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


# ===========================
    # Configuration
    # ===========================
    NUM_PARTICLES = 56000 #1000000
    OUTPUT_FILE = "part.txt"
    PARTICLE_DENSITY = 1000.0  # constant across all particles

    p = Particles(NUM_PARTICLES, OUTPUT_FILE)
    p.sample_diameter(method="uniform", choices =
                      [1E-4, 2.5E-4, 5E-4, 7.5E-4, 
                       1E-3, 2E-3, 3E-3, 4E-3, 5E-3, 
                       6E-3, 7E-3, 8E-3, 9E-3, 1E-2],)
    # p.sample_position(method="box", xmin=-40.0, xmax=40.0, ymin=-87.0, ymax=-7.0, zmin=-83.0, zmax=-3.0)
    p.sample_position(method="stl", stl_file="../honglin/honglin_setup/stl/inlet.stl", normal_offset=10.0, scale=0.7, flip_normal=True)    #False  
    p.generate()
    p.save_vtk()
    plot_distributions(OUTPUT_FILE)
