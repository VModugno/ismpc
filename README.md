# Description
This is IS-MPC, a framework for humanoid gait generation. The main reference is in this paper https://ieeexplore.ieee.org/document/8955951

# Implementation
You need a Python installation and some dependencis. If using PIP, you can run the following
```
pip install dartpy casadi scipy matplotlib
```
alternatively you can use the environment.yaml file to install the necessary dependancies for this code

```
mamba env create -f environment.yaml
```
an then you can activate the environment by doing:
```
conda activate biped_env
```

To run the simulation
```
python simulation.py
```
