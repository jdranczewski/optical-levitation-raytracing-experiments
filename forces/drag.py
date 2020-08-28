"""
Velocity-dependent drag

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""
# parameters: coefficient
from numpy import concatenate, zeros


def factory(config, params):
    coeff = params["coefficient"]

    def drag(state, t):
        return concatenate((-coeff * state[3:6], zeros(3)))

    return drag
