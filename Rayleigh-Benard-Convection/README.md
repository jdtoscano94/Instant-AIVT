# Artificial-Intelligence-Velocimetry-Thermometry (AIVT)

This is the official implementation of the paper:  
**"AIVT: Inference of turbulent thermal convection from measured 3D velocity data by physics-informed Kolmogorov-Arnold Networks".**

We propose the **Artificial Intelligence Velocimetry-Thermometry (AIVT)** method to reconstruct a continuous and differentiable representation of temperature and velocity in turbulent convection from measured 3D velocity data. AIVT is based on **physics-informed Kolmogorov-Arnold Networks (cKANs)** and is trained by optimizing a loss function that minimizes residuals of the velocity data, boundary conditions, and governing equations. 

We apply AIVT to a **unique dataset** containing simultaneously measured 3D temperature and velocity data of Rayleigh-Bénard convection, obtained through a combination of **Particle Image Thermometry (PIT) and Lagrangian Particle Tracking (LPT)**. Unlike previous studies, our approach **directly compares machine learning results to true volumetric, simultaneous temperature and velocity measurements**. 

We demonstrate that AIVT can **reconstruct and infer continuous, instantaneous velocity and temperature fields and their gradients from sparse experimental data**, achieving a fidelity comparable to direct numerical simulations of turbulence. This provides a powerful new avenue for analyzing turbulence at **high Reynolds numbers**.



## References

Please cite our work as:

```bibtex
@article{toscano2024inferring,
  title={Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks},
  author={Toscano, Juan Diego and K{"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2407.15727},
  year={2024}
}