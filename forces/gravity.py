"""
Constant gravitational force for the optical levitation ray tracer.

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""
# parameters: g, mass
from numpy import array, concatenate, zeros


def factory(config, params):
    g = array(params["g"])
    m = params["mass"]

    def gravity(state, t):
        return concatenate((g*m, zeros(3)))

    return gravity
