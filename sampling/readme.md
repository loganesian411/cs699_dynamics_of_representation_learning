This is the starter code for homework 2 (Sampling). 

This provides some utility functions for getting started and is based on JAX. 
JAX makes it easy to compute fn and its grad (see `main.py`), but you can use any other framework.
If you run into problem with other frameworks/library, please reach out to me. 
The discrete -> continuous density code provided in `utils/density.py` is just one way to convert discrete to continuous. 
Using any other approach for this is also fine. 

## Setup

- See `requirements.txt` for required packages. Higher version of packages should also work
- Tested with python3.8
- We have provided NPEET as a submodule. To fetch the NPEET code correctly, you may have to pass `--recurse_submodule` flag with git pull/clone command.

## How to use?
See `main.py` for suggestive use and homework document for more information. 

Note: The code should be executable with sampling as the root folder. 

## Note about using LSD
The git submodule requires some local changes in order to be used. After initializing and updating your local version of the git submodule off of the master LSD repo, please copy utils/lsd_toy.py to LSD/.
