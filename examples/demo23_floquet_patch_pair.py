"""DEMO 23: Free-space and waveguide-equivalent Floquet patch responses.

This example evaluates one grounded periodic patch in two environments:

1. ``free_space`` uses a normally incident Floquet plane wave.
2. ``waveguide_equivalent`` represents an air-filled rectangular waveguide's
   TE10 mode by changing the Floquet incidence angle at every frequency.

For a guide with broad-wall width ``a_wg`` the TE10 transverse wavenumber is
``kc = pi/a_wg``.  Decomposing TE10 into its two oblique plane-wave
constituents gives

    sin(theta(f)) = kc/k0 = fc10/f,
    beta10(f)     = k0*cos(theta(f)).

The default guide is 30 mm wide, corresponding to two 15 mm unit cells across
its broad wall.  Incidence is in the x-z plane (phi=0 degrees), and Floquet
mode 1 is S-polarized with E along y, matching TE10.  For this centered,
mirror-symmetric patch the +theta and -theta constituents have the same
co-polar reflection coefficient; asymmetric cells should check both signs.

Important implementation detail
-------------------------------
EMerge currently stores one scan angle on the shared periodic-boundary and
Floquet-port objects.  The waveguide-equivalent case therefore solves and
post-processes one frequency before changing the angle.  The normal-incidence
case has a fixed angle and may solve frequency jobs concurrently.

Both results are analytically de-embedded from the Floquet port to the patch
plane.  The script writes a CSV table, JSON metadata and a comparison plot.

The default mesh is a medium-cost research setting established between a
3,217-tetrahedron coarse mesh and an 11,879-tetrahedron refined mesh for this
geometry.  Exact tetrahedron count depends on the mesher version.

Typical invocation from the repository root::

    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src python \
      examples/demo23_floquet_patch_pair.py --workers 2

``--workers`` affects the fixed-angle free-space sweep only.  Keep
``--frequency-groups`` near the worker count to bound peak RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import emerge as em
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MM = 1e-3
GHZ = 1e9
C0 = em.C0
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "floquet_patch_pair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare normal-incidence and TE10-equivalent oblique-Floquet "
            "responses of one periodic grounded patch."
        )
    )
    parser.add_argument("--fmin-ghz", type=float, default=8.0)
    parser.add_argument("--fmax-ghz", type=float, default=12.0)
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent frequency jobs for the fixed-angle free-space sweep.",
    )
    parser.add_argument(
        "--frequency-groups",
        type=int,
        default=2,
        help="Matrices retained per free-space batch; 2 bounds RAM well.",
    )
    parser.add_argument("--resolution", type=float, default=0.20)
    parser.add_argument("--mesh-scale", type=float, default=2.75)
    parser.add_argument("--patch-edge-size-mm", type=float, default=0.30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not 0.0 < args.fmin_ghz < args.fmax_ghz:
        raise ValueError("Require 0 < fmin < fmax.")
    if args.points < 2:
        raise ValueError("At least two frequency points are required.")
    if args.workers < 1 or args.frequency_groups < 1:
        raise ValueError("workers and frequency-groups must be positive.")
    if min(args.resolution, args.mesh_scale, args.patch_edge_size_mm) <= 0:
        raise ValueError("Mesh settings must be positive.")


def build_and_solve(
    args: argparse.Namespace, waveguide_equivalent: bool
) -> dict[str, object]:
    """Build one periodic cell and solve one of the two excitation laws."""
    cell_size = 15.0 * MM
    patch_size = 6.0 * MM
    substrate_height = 1.6 * MM
    copper_thickness = 0.035 * MM
    air_height = 7.5 * MM
    guide_width = 2.0 * cell_size

    z_ground = copper_thickness
    z_patch = z_ground + substrate_height
    z_port = z_patch + air_height
    frequencies = np.linspace(
        args.fmin_ghz * GHZ, args.fmax_ghz * GHZ, args.points
    )

    substrate_material = em.Material(
        er=4.3,
        tand=0.025,
        color="#217627",
        opacity=0.35,
        name="FR-4",
    )
    copper = em.Material(
        er=1.0,
        ur=1.0,
        cond=5.8e7,
        color="EMERGE-COPPER",
        opacity=1.0,
        _metal=True,
        name="Copper",
    )

    label = "WaveguideEquivalent" if waveguide_equivalent else "FreeSpace"
    with em.Simulation(f"FloquetPatch{label}") as model:
        model.check_version("2.8.0")
        model.set_solver(em.EMSolver.SUPERLU)

        cell = em.RectCell(cell_size, cell_size)
        model.set_periodic_cell(cell)
        substrate = cell.volume(z_ground, z_patch).set_material(substrate_material)
        air = cell.volume(z_patch, z_port).background()
        ground = cell.port_face(z_ground).assign_thickness(copper_thickness)
        patch = em.geo.XYPlate(
            patch_size,
            patch_size,
            position=(0.0, 0.0, z_patch),
            alignment=em.CENTER,
        ).assign_thickness(copper_thickness)
        ground.set_material(copper)
        patch.set_material(copper)
        port_face = cell.port_face(z_port)

        # Mesh against the shortest wavelength, then reuse this mesh everywhere.
        model.mw.set_frequency(float(frequencies[-1]))
        model.mw.set_resolution(args.resolution)
        model.mw.set_basis_space(2, complete=False)
        model.commit_geometry()
        model.mesher.set_domain_size(
            substrate, 0.55 * args.mesh_scale * MM
        )
        model.mesher.set_domain_size(air, 1.8 * args.mesh_scale * MM)
        model.mesher.set_boundary_size(patch, args.patch_edge_size_mm * MM)
        model.mesher.set_face_size(port_face, 1.5 * args.mesh_scale * MM)

        start = perf_counter()
        model.generate_mesh()
        mesh_seconds = perf_counter() - start
        tetrahedra = int(model.mesh.n_tets)

        model.mw.bc.floquet_port(port_face, 1)
        model.mw.bc.ThinConductor(
            em.select(patch), material=copper, thickness=copper_thickness
        )
        model.mw.bc.SurfaceImpedance(
            em.select(ground), material=copper, thickness=copper_thickness
        )

        if not waveguide_equivalent:
            model.mw.set_frequency(frequencies)
            cell.set_scanangle(0.0, 0.0)
            start = perf_counter()
            data = model.mw.run_sweep(
                parallel=args.workers > 1,
                n_workers=args.workers,
                frequency_groups=args.frequency_groups,
            )
            solve_seconds = perf_counter() - start
            grid = data.scalar.slice_set(0, None, sort_by="freq").grid
            solved_freq = np.asarray(grid.freq)
            raw = np.asarray(grid.S(1.1, 1.1))
            theta_deg = np.zeros_like(solved_freq)
            beta = 2.0 * np.pi * solved_freq / C0
            point_seconds: list[float] | None = None
            workers = args.workers
        else:
            kc10 = np.pi / guide_width
            raw_values: list[complex] = []
            theta_values: list[float] = []
            beta_values: list[float] = []
            point_seconds = []

            for frequency in frequencies:
                k0 = 2.0 * np.pi * frequency / C0
                if k0 <= kc10:
                    raise ValueError(
                        f"TE10 is below cutoff at {frequency/GHZ:.3f} GHz."
                    )
                theta = float(np.arcsin(kc10 / k0))
                model.mw.set_frequency(float(frequency))
                cell.set_scanangle(float(np.degrees(theta)), 0.0)

                start = perf_counter()
                data = model.mw.run_sweep(
                    parallel=False, n_workers=1, frequency_groups=1
                )
                point_seconds.append(perf_counter() - start)
                latest = data.scalar.slice_set(
                    data.scalar.n - 1, None, sort_by="freq"
                ).grid
                raw_values.append(
                    complex(np.asarray(latest.S(1.1, 1.1)).reshape(-1)[-1])
                )
                theta_values.append(float(np.degrees(theta)))
                beta_values.append(float(k0 * np.cos(theta)))

            solved_freq = frequencies
            raw = np.asarray(raw_values)
            theta_deg = np.asarray(theta_values)
            beta = np.asarray(beta_values)
            solve_seconds = float(np.sum(point_seconds))
            workers = 1

    return {
        "freq": solved_freq,
        "theta_deg": theta_deg,
        "beta": beta,
        "s11_raw": raw,
        "s11": raw * np.exp(2j * beta * air_height),
        "tetrahedra": tetrahedra,
        "mesh_seconds": mesh_seconds,
        "solve_seconds": solve_seconds,
        "point_seconds": point_seconds,
        "workers": workers,
        "guide_width_mm": guide_width / MM,
        "te10_cutoff_ghz": C0 / (2.0 * guide_width) / GHZ,
    }


def unwrap_phase_deg(values: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.angle(values)))


def write_results(
    args: argparse.Namespace,
    free_space: dict[str, object],
    waveguide: dict[str, object],
) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(free_space["freq"])
    wg_freq = np.asarray(waveguide["freq"])
    if not np.allclose(freq, wg_freq, rtol=1e-12):
        raise RuntimeError("The two scenarios returned different frequency grids.")

    fs = np.asarray(free_space["s11"])
    wg = np.asarray(waveguide["s11"])
    fs_phase = unwrap_phase_deg(fs)
    wg_phase = unwrap_phase_deg(wg)
    theta = np.asarray(waveguide["theta_deg"])

    csv_path = output / "responses.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frequency_hz",
                "frequency_ghz",
                "te10_equivalent_theta_deg",
                "free_space_s11_real",
                "free_space_s11_imag",
                "free_space_magnitude_db",
                "free_space_phase_deg",
                "waveguide_equivalent_s11_real",
                "waveguide_equivalent_s11_imag",
                "waveguide_equivalent_magnitude_db",
                "waveguide_equivalent_phase_deg",
            ]
        )
        for i, frequency in enumerate(freq):
            writer.writerow(
                [
                    frequency,
                    frequency / GHZ,
                    theta[i],
                    fs[i].real,
                    fs[i].imag,
                    20.0 * np.log10(max(abs(fs[i]), 1e-15)),
                    fs_phase[i],
                    wg[i].real,
                    wg[i].imag,
                    20.0 * np.log10(max(abs(wg[i]), 1e-15)),
                    wg_phase[i],
                ]
            )

    metadata = {
        "purpose": (
            "One-design comparison of normal free-space and frequency-dependent "
            "TE10-equivalent Floquet excitation"
        ),
        "geometry_mm": {
            "cell": 15.0,
            "square_patch": 6.0,
            "substrate_height": 1.6,
            "air_to_port": 7.5,
            "assumed_waveguide_width": waveguide["guide_width_mm"],
        },
        "materials": {
            "substrate_er": 4.3,
            "substrate_tand": 0.025,
            "copper_conductivity_s_per_m": 5.8e7,
            "copper_thickness_mm": 0.035,
        },
        "sweep": {
            "fmin_ghz": args.fmin_ghz,
            "fmax_ghz": args.fmax_ghz,
            "points": args.points,
            "te10_cutoff_ghz": waveguide["te10_cutoff_ghz"],
            "polarization": "Floquet mode 1 (S), E along y, phi=0 deg",
            "reference_plane": "patch plane after analytic de-embedding",
        },
        "mesh": {
            "resolution": args.resolution,
            "mesh_scale": args.mesh_scale,
            "patch_edge_size_mm": args.patch_edge_size_mm,
            "free_space_tetrahedra": free_space["tetrahedra"],
            "waveguide_equivalent_tetrahedra": waveguide["tetrahedra"],
        },
        "performance": {
            "free_space_workers_cores": free_space["workers"],
            "free_space_mesh_seconds": free_space["mesh_seconds"],
            "free_space_sweep_seconds": free_space["solve_seconds"],
            "waveguide_equivalent_workers_cores": waveguide["workers"],
            "waveguide_equivalent_mesh_seconds": waveguide["mesh_seconds"],
            "waveguide_equivalent_sweep_seconds": waveguide["solve_seconds"],
            "waveguide_equivalent_per_point_seconds": waveguide["point_seconds"],
            "ram_note": (
                "Peak RSS is process/environment dependent; measure with "
                "/usr/bin/time -v. frequency-groups bounds concurrent matrices."
            ),
        },
        "limitations": [
            "This is a TE10 angle-equivalent infinite-periodic surrogate, not a finite metallic guide.",
            "A centered symmetric cell is assumed when one oblique constituent represents both signs.",
            "Higher Floquet orders and physical waveguide higher modes are not included.",
        ],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    axes[0].plot(freq / GHZ, fs_phase, label="free space: normal Floquet")
    axes[0].plot(freq / GHZ, wg_phase, label="TE10-equivalent Floquet")
    axes[0].set_ylabel("De-embedded S11 phase [deg]")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(freq / GHZ, 20.0 * np.log10(np.maximum(abs(fs), 1e-15)))
    axes[1].plot(freq / GHZ, 20.0 * np.log10(np.maximum(abs(wg), 1e-15)))
    axes[1].set_xlabel("Frequency [GHz]")
    axes[1].set_ylabel("De-embedded |S11| [dB]")
    axes[1].grid(True)
    fig.tight_layout()
    plot_path = output / "responses.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved {csv_path}")
    print(f"Saved {metadata_path}")
    print(f"Saved {plot_path}")
    print(
        "Free space: "
        f"{free_space['tetrahedra']:,} tets, {free_space['workers']} core(s), "
        f"{free_space['solve_seconds']:.2f} s"
    )
    print(
        "TE10-equivalent: "
        f"{waveguide['tetrahedra']:,} tets, 1 core, "
        f"{waveguide['solve_seconds']:.2f} s"
    )


def main() -> None:
    args = parse_args()
    validate(args)
    print("Solving normal-incidence free-space Floquet response...")
    free_space = build_and_solve(args, waveguide_equivalent=False)
    print("Solving frequency-dependent TE10-equivalent Floquet response...")
    waveguide = build_and_solve(args, waveguide_equivalent=True)
    write_results(args, free_space, waveguide)


if __name__ == "__main__":
    main()
