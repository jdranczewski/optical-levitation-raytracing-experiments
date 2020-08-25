"""
Force for a central potential.

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""
from numpy import array, concatenate, zeros


def factory(config, params):
    origin = array(params["origin"])
    coeff = params["coefficient"]

    def central(state, t):
        pos = state[:3] - origin
        return concatenate((-coeff * pos / (pos[0]**2 + pos[1]**2 + pos[2]**2)**(3/2), zeros(3)))

    return central
