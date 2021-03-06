# optical-levitation-raytracing-experiments

Created during an Undergraduate Research Experience Programme placement at
Imperial College London 2020 by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

# Setup
A virtual environment is recommended, but not strictly required. The non-standard
packages used are listed in `requirements.txt`. Quite a few of these probably
come with a standard Anaconda installation, but not necessarily all of them.

To install all the requirements automatically, just run:

`pip install -r requirements.txt`

Now, this command will slightly vary depending on your particular Python setup.
This margin is to small to delve into the
[magical land of Python environments](https://xkcd.com/1987/),
but as a general rule, if you're in a virtual environment `pip` should be fine.
If not, and you have more than one Python, you may need to use `pip3`.
Or, if that doesn't work, maybe `pythom -m pip` will do the trick?

# Structure of this repository

## `simulate.py`
The most important file for most use cases. It allows for running configuration
files and simulating the behaviour of a given target.

The behaviour of this script largely depends on the chosen configuration file,
which is better described in `configs\config_example.yaml`. In general
it will run a simulation and the display / save the results in the `outputs`
directory.

**The script runs the `config.yaml` file in its directory by default.**

To run arbitrary files, you need to pass them as command line arguments.
For example:

`> python simulate.py configs/config_example.yaml`

The output directory includes a `README.md` file explaining the output file
format.


## `output-inspector.ipynb`
A Jupyter Notebook that can be used to analyse output files produced by
`simulate.py`. It includes a few convenient utilities, like orientation
visualisation, graphing the ray tracer's output at any point in time,
and quiver plotting.

It can also be used as a guide to how the data can be analysed.


## `ptracer.py`
The ray tracer library. This can be used on its own to investigate particular
situations, which is demonstrated in `ptracer-examples.ipynb` and, in more
detail, in `various-experiments\ptracer-experiments.ipynb`.

## `jit_methods`
This file includes some computationally heavy or often-called functions,
which are then compiled by numba.

## `notes.pdf`
A set of handwritten notes containing most of the derivations for maths used
in the code.

## Directories
* `configs` contains example configuration files and is a good place to put
  your config files as well.
* `forces` contains python files describing the various forces that can be
  used in `simulate.py`.
* `objs` contains some .obj files, which describe 3D meshes. Note that some
  of them do not contain information about vertex normal and cannot be used
  in the SmoothMeshTO class.
* `output` is where `simulate.py` puts output directories.
* `various-experiments` contains code and Jupyter Notebooks used and created
  during the project to test various idea and implementations.

**Some of these directories contain `README.md` files with further information.

# Some references
* [Ray-triangle interestcion algorithm in 3D](http://webserver2.tecgraf.puc-rio.br/~mgattass/cg/trbRR/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf) -
this describes the Möller–Trumbore algorithm implementation used here.
* [A new Leapfrog scheme for rotational motion in 3D](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.5165) -
this paper is the source of the Ordinary Differential Equations used here for
rotational dynamics.
* [Visualizing quaternions](https://eater.net/quaternions) -
a great deep-dive into what quaternions are and how they relate to 3D rotations.
