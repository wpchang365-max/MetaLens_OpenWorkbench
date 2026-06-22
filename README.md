# MetaLens Open Workbench

**MetaLens Open Workbench** is an open-source Python workbench for metalens design, optical-field propagation simulation, projection-lithography imaging evaluation, optimization, GPU/CPU computation, meta-atom database integration, layout export, and reproducible research workflows.

This project provides a graphical research environment and reusable Python modules for rapidly prototyping computational metalens designs. It is intended for researchers, students, and developers who need an extensible tool for exploring phase profiles, point-spread functions, lithography-oriented imaging metrics, tolerance analysis, and data-driven design workflows.

---

## Features

* Metalens phase-profile design and visualization
* Optical-field propagation simulation
* Fresnel and angular-spectrum propagation models
* Fast Debye-style approximation for focusing analysis
* Point-spread function calculation and evaluation
* FWHM, sidelobe ratio, and intensity-profile analysis
* Projection-lithography imaging evaluation
* Critical dimension and NILS-related analysis
* Particle swarm optimization and marine predator-inspired optimization
* Historical-data-assisted design workflow
* CPU computation with optional GPU acceleration
* Optional CUDA acceleration through CuPy
* Optional OpenCL acceleration through PyOpenCL
* Meta-atom database import and interpolation
* GDS/DXF layout export for nanofabrication workflows
* JSON, CSV, image, and project-data export
* Graphical user interface for interactive research use

---

## Project Scope

This software is designed as an open research tool and computational prototype platform for metalens and computational-optics studies.

It is **not** intended to replace rigorous full-wave electromagnetic solvers such as FDTD, FEM, or RCWA. The built-in propagation models are mainly based on scalar Fourier optics, Fresnel propagation, angular-spectrum propagation, and fast approximation methods. For accurate nano-structure responses, meta-atom transmission coefficients, fabrication errors, and material dispersion, users are encouraged to combine this software with full-wave simulations or experimental calibration data.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/MetaLens-Open-Workbench.git
cd MetaLens-Open-Workbench
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the basic dependencies can be installed manually:

```bash
pip install numpy pillow
```

For optional CUDA GPU acceleration:

```bash
pip install cupy
```

For optional OpenCL GPU acceleration:

```bash
pip install pyopencl
```

---

## Quick Start

Run the graphical interface:

```bash
python run_gpu_lens_modern.py
```

Alternatively, run the package module:

```bash
python -m gpu_lens_modern
```

The software will automatically use available CPU or GPU backends depending on the installed dependencies and the local hardware environment.

---

## Example: Basic Python Usage

The project can also be used as a Python research toolkit. A typical workflow is shown below:

```python
import numpy as np

# Define basic optical parameters
wavelength = 405e-9
focal_length = 10e-6
lens_radius = 5e-6
grid_size = 512

# Build a computational grid
x = np.linspace(-lens_radius, lens_radius, grid_size)
y = np.linspace(-lens_radius, lens_radius, grid_size)
X, Y = np.meshgrid(x, y)

# Example ideal metalens phase profile
phase = -2 * np.pi / wavelength * (
    np.sqrt(X**2 + Y**2 + focal_length**2) - focal_length
)

# Convert phase to a wrapped 0-2pi distribution
phase_wrapped = np.mod(phase, 2 * np.pi)

print("Metalens phase profile generated.")
print("Grid size:", phase_wrapped.shape)
```

---

## Example: GDS Layout Export Workflow

A simplified layout-generation workflow is shown below. The actual export function may depend on the installed layout backend and the project module structure.

```python
# Example pseudo-workflow for layout generation

# 1. Generate or import a phase profile
# 2. Map the phase profile to meta-atom geometry
# 3. Convert the geometry map to layout polygons
# 4. Export the final structure to GDS or DXF

phase_map = "phase_profile.npy"
meta_atom_database = "meta_atom_database.csv"
output_file = "metalens_layout.gds"

print("Input phase map:", phase_map)
print("Meta-atom database:", meta_atom_database)
print("Output layout:", output_file)
```

---

## Typical Research Workflow

1. Define the optical design target, such as wavelength, focal length, numerical aperture, and lens aperture.
2. Generate the target phase profile of the metalens.
3. Import or construct a meta-atom database.
4. Map the required phase distribution to physical meta-atom geometries.
5. Simulate the optical field near the focal plane.
6. Evaluate FWHM, sidelobe ratio, peak intensity, and imaging quality.
7. Optimize the design parameters if necessary.
8. Export the final design data, simulation results, and layout files.

---

## Repository Structure

A typical project structure is:

```text
MetaLens-Open-Workbench/
├── gpu_lens_modern/
│   ├── __init__.py
│   ├── core/
│   ├── gui/
│   ├── optimization/
│   ├── propagation/
│   ├── layout/
│   └── utils/
├── run_gpu_lens_modern.py
├── requirements.txt
├── README.md
└── LICENSE
```

The actual structure may vary depending on the released version.

---

## Dependencies

The core version requires:

* Python 3.8 or later
* NumPy
* Pillow

Optional dependencies include:

* CuPy for CUDA acceleration
* PyOpenCL for OpenCL acceleration
* Matplotlib for additional visualization
* SciPy for extended numerical routines
* gdspy or gdstk for GDS layout export

---

## GPU Acceleration

The software supports optional GPU acceleration. If a compatible CUDA environment and CuPy are available, CUDA-based computation can be used. If PyOpenCL and a supported OpenCL device are available, OpenCL acceleration can also be used.

If no GPU backend is detected, the software will fall back to CPU computation.

---

## Data Export

The software supports exporting research data in multiple formats, including:

* JSON project files
* CSV numerical data
* PNG/TIFF images
* NumPy arrays
* GDS/DXF layout files
* Simulation result summaries

These export functions are intended to support reproducible research, data sharing, and further post-processing.

---

## Limitations

Please note the following limitations before using this project:

* The built-in optical propagation models are mainly approximate models.
* The software does not replace rigorous full-wave electromagnetic simulation.
* Meta-atom responses should be verified by FDTD, FEM, RCWA, or experimental measurements.
* Fabrication errors, sidewall angle, material dispersion, and substrate effects may need additional calibration.
* GPU acceleration depends on the local hardware, driver, and Python environment.
* Some advanced functions may require optional dependencies.

---

## Recommended Use Cases

This project is suitable for:

* Metalens phase-profile design
* Computational-optics teaching and demonstration
* Rapid prototyping of metalens concepts
* Projection-lithography imaging analysis
* Point-spread function evaluation
* Optimization-method testing
* Research-data visualization and export
* Layout-preparation workflow development

---

## Citation

If this project is useful for your research, please consider citing it as:

```bibtex
@software{metalens_open_workbench,
  title  = {MetaLens Open Workbench: An Open-Source Python Workbench for Metalens Design and Optical Simulation},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/your-username/MetaLens-Open-Workbench}
}
```

Please replace the author name and repository URL with the correct information before publication.

---

## Contributing

Contributions are welcome. You may contribute by:

* Reporting bugs
* Suggesting new features
* Improving documentation
* Adding examples
* Optimizing numerical performance
* Extending GPU support
* Improving layout export functions
* Adding new optical propagation or optimization models

Before submitting a pull request, please make sure that the code is readable, documented, and tested with a minimal example.

---

## License

This project is released under the MIT License.

You are free to use, modify, distribute, and build upon this project under the terms of the license. Please see the `LICENSE` file for details.

---

## Disclaimer

This software is provided for academic research, education, and open-source development. The authors make no warranty regarding the accuracy, completeness, or suitability of the software for any specific application. Users are responsible for validating all simulation results before using them in scientific publications, fabrication processes, or engineering applications.
