import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from tqdm import tqdm

from read_part import write_vtp

DIAMETERS = [1E-4, 2.5E-4, 5E-4, 7.5E-4, 1E-3, 2E-3, 3E-3,
             4E-3, 5E-3, 6E-3, 7E-3, 8E-3, 9E-3, 1E-2]


class Particles:
    '''Sample particle initial conditions and write them for the solver.'''

    def __init__(self, n_particles, output_file, density = 1000.0, seed = None):
        self.n_particles = n_particles
        self.output_file = output_file
        self.density = density
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.positions = None
        self.diameters = None

    def sample_position(self, method, **kwargs):
        if method == "box":
            self.positions = np.column_stack([
                self.rng.uniform(kwargs[f"{a}min"], kwargs[f"{a}max"], self.n_particles)
                for a in "xyz"])

        elif method == "sphere":
            self.positions = self._sample_sphere(
                center=np.asarray(kwargs["center"], float),
                radius=float(kwargs["radius"]),
                stl_file=kwargs.get("stl_file"),
                clearance=kwargs.get("clearance", 0.0),
                inside_stl=kwargs.get("inside_stl", False),
                profile=kwargs.get("profile", "uniform"),
                r_min=kwargs.get("r_min", 0.0))

        elif method == "stl":
            stl_file = kwargs["stl_file"]
            normal_offset = kwargs.get("normal_offset", 3.0e-4)
            tangential_jitter = kwargs.get("tangential_jitter", 1.0e-5)
            flip_normal = kwargs.get("flip_normal", False)
            scale = kwargs.get("scale", 1.0)

            mesh = trimesh.load_mesh(stl_file, process=False)
            if mesh.is_empty:
                raise RuntimeError(f"Loaded mesh is empty: {stl_file}")

            points, face_idx = trimesh.sample.sample_surface(
                mesh, self.n_particles, seed=self.seed)
            if scale != 1.0:
                center = mesh.vertices.mean(axis=0)
                points = center + scale * (points - center)

            # step off the surface so particles start inside the flow
            sign = 1.0 if flip_normal else -1.0
            points = points + sign * normal_offset * mesh.face_normals[face_idx]
            if tangential_jitter > 0.0:
                points = points + self.rng.normal(0.0, tangential_jitter, points.shape)
            self.positions = points

        else:
            raise ValueError(f"Unknown position method: {method}")

    def _ball(self, n, center, radius, profile = "uniform", r_min = 0.0):
        '''Isotropic direction, radius drawn according to `profile`.

        uniform        : constant density, so the count per unit radius grows as r^2
        inverse_square : density ~ 1/r^2, so every radius gets the same count
        '''
        d = self.rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        u = self.rng.uniform(size=n)

        if profile == "uniform":
            r = (r_min**3 + (radius**3 - r_min**3) * u) ** (1 / 3)
        elif profile == "inverse_square":
            r = r_min + (radius - r_min) * u
        else:
            raise ValueError(f"Unknown radial profile: {profile}")

        return center + d * r[:, None]

    def _sample_sphere(self, center, radius, stl_file, clearance, inside_stl,
                       profile = "uniform", r_min = 0.0, max_passes = 20):
        '''Sample a sphere, keeping only points clear of an STL surface.

        The surface must be closed: the test is the signed distance to it, which
        has no meaning on an open one. Positive is outside, so `clearance` is the
        margin a particle must keep from the surface.

        The radial profile is applied before the surface test, so wherever the
        sphere overlaps the surface the delivered count is the profile times the
        locally allowed fraction.
        '''
        ball = lambda n: self._ball(n, center, radius, profile, r_min)
        if stl_file is None:
            return ball(self.n_particles)

        surf = pv.read(stl_file).clean()
        open_edges = surf.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False).n_cells
        if open_edges:
            raise ValueError(
                f"{stl_file} is not closed ({open_edges} open edges), so inside "
                "and outside are undefined. Merge the pieces into one closed "
                "surface first.")

        sign = -1.0 if inside_stl else 1.0
        kept = []
        have = 0
        accept = 0.5                      # refined from the first pass
        for _ in range(max_passes):
            n = int((self.n_particles - have) / max(accept, 1e-3) * 1.2) + 100
            n = min(n, 10 * self.n_particles + 1000)   # a low rate must not explode the batch
            pts = ball(n)
            dist = pv.PolyData(pts).compute_implicit_distance(surf)["implicit_distance"]
            good = pts[sign * dist > clearance]

            accept = len(good) / n
            kept.append(good)
            have += len(good)
            if not have:
                break                                  # nothing in the sphere is allowed
            if have >= self.n_particles:
                out = np.vstack(kept)[:self.n_particles]
                print(f"sphere sampling: kept {self.n_particles} of "
                      f"{sum(len(k) for k in kept)} sampled, acceptance {accept:.3f}")
                return out

        raise RuntimeError(
            f"only found {have} of {self.n_particles} points in {max_passes} passes "
            f"(acceptance {accept:.4f}); the sphere may barely reach the allowed region")

    def sample_diameter(self, method, **kwargs):
        if method == "log-normal":
            self.diameters = 1e-4 * self.rng.lognormal(
                mean=kwargs.get("mean", 0.0),
                sigma=kwargs.get("sigma", 1.0),
                size=self.n_particles)

        elif method == "uniform":
            choices = kwargs.get("choices")
            if choices is None:
                raise ValueError("The 'uniform' method requires a 'choices' argument.")
            self.diameters = self.rng.choice(choices, size=self.n_particles)

        else:
            raise ValueError(f"Unknown diameter method: {method}")

    def _check(self, caller):
        if self.positions is None or self.diameters is None:
            raise RuntimeError(
                f"Call sample_position() and sample_diameter() before {caller}().")

    def generate(self):
        '''Write the solver input: diameter, density, position.'''
        self._check("generate")
        with open(self.output_file, "w") as f:
            for i in tqdm(range(self.n_particles)):
                row = [self.diameters[i], self.density] + self.positions[i].tolist()
                f.write("\t".join(f"{val:.6f}" for val in row) + "\n")

    def save_vtk(self, vtk_file = None):
        self._check("save_vtk")
        if vtk_file is None:
            vtk_file = self.output_file.rsplit(".", 1)[0] + ".vtk"

        n = self.n_particles
        with open(vtk_file, "w") as f:
            f.write("# vtk DataFile Version 2.0\nParticles\nASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {n} float\n")
            for p in self.positions:
                f.write(f"{p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
            f.write(f"VERTICES {n} {2 * n}\n")
            for i in range(n):
                f.write(f"1 {i}\n")
            f.write(f"POINT_DATA {n}\n")
            f.write("SCALARS diameter float 1\nLOOKUP_TABLE default\n")
            for d in self.diameters:
                f.write(f"{d:.6e}\n")
            f.write("SCALARS density float 1\nLOOKUP_TABLE default\n")
            for _ in range(n):
                f.write(f"{self.density:.6e}\n")

        print(f"Wrote VTK file: {vtk_file}")


def plot_distributions(file_path, fileName = "particle_distribution.png"):
    data = np.loadtxt(file_path, delimiter="\t")
    diameters, positions = data[:, 0], data[:, 2:5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(diameters, bins=30, density=True, color="gray")
    ax1.set(title="PDF of Diameter", xlabel="Diameter", ylabel="Probability Density")

    for i, label in enumerate("xyz"):
        ax2.hist(positions[:, i], bins=30, density=True, alpha=0.6, label=label)
    ax2.set(title="PDF of Initial Position Components",
            xlabel="Position", ylabel="Probability Density")
    ax2.legend()

    fig.savefig(fileName, bbox_inches="tight")
    plt.close(fig)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-n", "--count", type=int, default=56000)
    common.add_argument("-o", "--output", default="part",
                        help="output stem: writes <stem>.txt and <stem>.vtp")
    common.add_argument("--diameters", type=float, nargs="+", default=DIAMETERS,
                        help="diameters to draw uniformly from")
    common.add_argument("--density", type=float, default=1000.0)
    common.add_argument("--seed", type=int)
    common.add_argument("--plot", action="store_true",
                        help="also write <stem>_distribution.png")

    ap = argparse.ArgumentParser(description="Generate particle initial conditions.")
    sub = ap.add_subparsers(dest="method", required=True)

    s = sub.add_parser("sphere", parents=[common], help="fill a sphere")
    s.add_argument("--center", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    s.add_argument("--radius", type=float, required=True)
    s.add_argument("--stl", help="closed surface to keep particles outside of")
    s.add_argument("--clearance", type=float, default=0.0,
                   help="minimum distance from the STL surface")
    s.add_argument("--inside", action="store_true", help="keep inside the STL instead")
    s.add_argument("--profile", choices=["uniform", "inverse_square"], default="uniform")
    s.add_argument("--r-min", type=float, default=0.0)

    t = sub.add_parser("stl", parents=[common], help="sample an STL surface")
    t.add_argument("stl_file")
    t.add_argument("--normal-offset", type=float, default=3.0e-4)
    t.add_argument("--scale", type=float, default=1.0)
    t.add_argument("--flip-normal", action="store_true")
    t.add_argument("--jitter", type=float, default=1.0e-5)

    b = sub.add_parser("box", parents=[common], help="fill a box")
    b.add_argument("--min", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    b.add_argument("--max", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))

    args = ap.parse_args()

    p = Particles(args.count, args.output + ".txt", density=args.density, seed=args.seed)
    p.sample_diameter(method="uniform", choices=args.diameters)
    if args.method == "sphere":
        p.sample_position(method="sphere", center=args.center, radius=args.radius,
                          stl_file=args.stl, clearance=args.clearance,
                          inside_stl=args.inside, profile=args.profile, r_min=args.r_min)
    elif args.method == "stl":
        p.sample_position(method="stl", stl_file=args.stl_file,
                          normal_offset=args.normal_offset, scale=args.scale,
                          flip_normal=args.flip_normal, tangential_jitter=args.jitter)
    else:
        p.sample_position(method="box", **{f"{a}min": lo for a, lo in zip("xyz", args.min)},
                          **{f"{a}max": hi for a, hi in zip("xyz", args.max)})

    p.generate()
    write_vtp(pd.DataFrame(p.positions, columns=["x", "y", "z"])
              .assign(diam=p.diameters, density=p.density), args.output + ".vtp")
    print(f"wrote {args.output}.txt and {args.output}.vtp")
    if args.plot:
        plot_distributions(args.output + ".txt", args.output + "_distribution.png")


if __name__ == "__main__":
    main()
