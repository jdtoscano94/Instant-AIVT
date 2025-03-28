# Instant Artificial-Intelligence-Velocimetry-Thermometry (AIVT)

This repository serves as a collection of code implementations for published and upcoming papers on Artificial Intelligence Velocimetry-Thermometry (AIVT). It will be continuously updated with new research developments and additional resources.

## Papers

Currently, this repository contains the code for the following paper:

1. **Toscano, J. D., Käufer, T., Wang, Z., Maxey, M., Cierpka, C., & Karniadakis, G. E. (2024).** Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks. *arXiv preprint arXiv:2407.15727.*  
   **Link:** [arXiv:2407.15727](https://arxiv.org/abs/2407.15727)

Additional papers and updates will be added to this repository as they become available.

## Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/jdtoscano94/Instant-AIVT.git
   cd YOUR_REPO_FOLDER
   ```

2. **Download the dataset**:  
   The dataset is available [here](https://drive.google.com/file/d/1HXJGubY4J2TN4DOuhFA5pu_6A2rw7FgM/view?usp=drive_link).

3. **Move the data to the correct directory**:  
   ```sh
   mv path_to_downloaded_data ../Data/Rayleigh-Benard-Convection/
   ```

4. **Run our models using the provided Jupyter notebooks**.  
   The repository includes results for the following models:
   - **cKAN** with **149k parameters**  
   - **MLP** with **151k parameters**  
   - **MLP** with **282k parameters**  

   Each notebook contains all the necessary code to **replicate the results** presented in the paper. The results were generated using **JAX**, ensuring efficient computation with accelerated hardware support.

**Note:** To run these notebooks, you need the source files located in `../Instant_AIVT`. These files should be automatically downloaded when the project is cloned. Ensure that all dependencies are installed before executing the notebooks.

## References

If you use this repository in your research, please cite our related work:

```bibtex

@article{anagnostopoulos2024residual,
  title={Residual-based attention in physics-informed neural networks},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={421},
  pages={116805},
  year={2024},
  publisher={Elsevier}
}

@article{anagnostopoulos2024learning,
  title={Learning in PINNs: Phase transition, total diffusion, and generalization},
  author={Anagnostopoulos, Sokratis J and Toscano, Juan Diego and Stergiopulos, Nikolaos and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2403.18494},
  year={2024}
}

@article{shukla2024comprehensive,
  title={A comprehensive and fair comparison between mlp and kan representations for differential equations and operator networks},
  author={Shukla, Khemraj and Toscano, Juan Diego and Wang, Zhicheng and Zou, Zongren and Karniadakis, George Em},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={431},
  pages={117290},
  year={2024},
  publisher={Elsevier}
}


@article{toscano2024inferring,
  title={Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks},
  author={Toscano, Juan Diego and K{"a}ufer, Theo and Wang, Zhibo and Maxey, Martin and Cierpka, Christian and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2407.15727},
  year={2024}
}

```
