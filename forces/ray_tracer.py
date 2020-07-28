"""
Force based on a ray tracer.

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""
import tdt2
from numpy import array
import matplotlib.pyplot as plt


def factory(config, params):
    def central(state, t):
        rf = getattr(tdt2, params["ray-factory"]["type"])(**params["ray-factory"]["params"])
        objects = []
        for i, obj in enumerate(params["objects"]):
            if obj["origin"]["type"] == "offset":
                origin = state[:2] + array(obj["origin"]["value"])
            else:
                origin = array(obj["origin"]["value"])
            objects.append(getattr(tdt2, obj["type"])(origin, **obj["params"]))
        # print(objects[0].origin)
        # print(t)
        scene = tdt2.Scene(rf, objects)
        scene.run()
        scene.propagate(5e-6)
        # fig, ax = plt.subplots()
        # scene.plot(ax, m_quiver=True)
        # plt.show()
        # print(scene.momentum)
        return scene.momentum*6.62607004e-34*1e9

    return central
