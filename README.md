# Neural Metamaterial Networks for Nonlinear Material Design
![image](img/teaser.png)\
[Yue Li](https://liyuesolo.github.io/), [Stelian Coros](https://crl.ethz.ch/people/coros/index.html), [Bernhard Thomaszewski](https://n.ethz.ch/~bthomasz/)\
ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2023)
### [Paper](https://arxiv.org/pdf/2309.10600.pdf) [Video](https://www.youtube.com/watch?v=NHLYxoZ2O_s)

## Code Structure

Projects/NeuralMetamaterial/
- [include & src] contains header and source files for the FEM solver
- [python] contains pretained models, code for training and inverse design
- [scripts] for generating training data
- [data] example data file for training

## Compile and Run
Simply execute the build.sh file


## Third Party Library Required

The version that is used to compile this code under Ubuntu 20.04 is attached as ThirdParty.zip

Libigl (https://libigl.github.io/) \
Gmsh (https://gmsh.info/) \
IPC Toolkit (https://ipc-sim.github.io/ipc-toolkit/) \
SuiteSparse (https://github.com/DrTimothyAldenDavis/SuiteSparse)

These folder should be in the same folder as the base folder.\
The versions that were used for this repo can be found [here](https://drive.google.com/file/d/1B55HXDseMPa6bjgzhKqjrdqMLtarY8gW/view?usp=drive_link).

## Third Party Attached

Tactile (https://github.com/isohedral/tactile) \
Clipper (https://github.com/AngusJohnson/Clipper2) \
SpectrA (https://spectralib.org/) \
Poisson Disk Sampling (http://www.cemyuksel.com/cyCodeBase/soln/poisson_disk_sampling.html)

## License
See the [LICENSE](LICENSE) file for license rights and limitations.

## Contact
Please reach out to me ([Yue Li](yueli.cg@gmail.com)) if you spot any bugs or having quesitons or suggestions!