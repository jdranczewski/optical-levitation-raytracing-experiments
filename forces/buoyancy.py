"""
Buoyant force for the optical levitation ray tracer.

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""
from numpy import array


def factory(config, params):
    g = array(params["g"])
    density = params["density"]
    volume = params["volume"]

    def buoyancy(state, t):
        return -g*density*volume

    return buoyancy
