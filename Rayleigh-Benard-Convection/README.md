# Artificial-Intelligence-Velocimetry-Thermometry-AIVT


We propose the Artificial Intelligence Velocimetry-Thermometry (AIVT) method to reconstruct a continuous and differentiable representation of the temperature and velocity in turbulent convection from measured 3D velocity data. AIVT is based on physics-informed Kolmogorov-Arnold Networks and trained by optimizing a loss function that minimizes residuals of the velocity data, boundary conditions, and governing equations. We apply AIVT to a new and unique set of simultaneously measured 3D temperature and velocity data of Rayleigh-Bénard convection, obtained by combining Particle Image Thermometry and Lagrangian Particle Tracking. This enables us, for the first time and unlike previous studies, to directly compare machine learning results to true volumetric, simultaneous temperature and velocity measurements. We demonstrate that AIVT can reconstruct and infer continuous, instantaneous velocity and temperature fields and their gradients from sparse experimental data at a fidelity comparable to direct numerical simulations of turbulence, providing an avenue for understanding turbulence at high Reynolds numbers.

### Instructions:

1. Clone the repository.

2. Please download the data from here: https://drive.google.com/file/d/1HXJGubY4J2TN4DOuhFA5pu_6A2rw7FgM/view?usp=drive_link

3. Move data to the folder ../Data/Rayleigh-Benard-Convection/

4. Run our models using the provided jupyter notebooks. We present results for:

- cKAN with 149k parameters
- MLP with 151k parameters
- MlP with 282k parameters

Each notebook has all the necesary code requireded to replicate the resuls in the paper

### References

Please cite our work as:

@article{toscano2024inferring,
  title={Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks},
  author={Toscano, Juan Diego and K{\"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2407.15727},
  year={2024}
}

