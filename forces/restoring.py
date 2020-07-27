"""
Force for a restoring force, linear with displacement.

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
    origin = array(params["origin"])
    coeff = params["coefficient"]

    def central(state, t):
        return -coeff * (state[:2] - origin)

    return central
