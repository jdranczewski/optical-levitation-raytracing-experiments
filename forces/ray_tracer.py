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
import ptracer
from numpy import array
import matplotlib.pyplot as plt


def factory(config, params):
    def ray_tracer_force(state, t):
        scene = make_scene(state, params)
        scene.run()
        # if state[2]>1.3e-5:
        # scene.propagate(1e-6)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scene.plot(ax, m_quiver=True, ray_kwargs={"c": "tab:blue"})
        # plt.show()
        # print(scene.momentum, state[:2])
        return scene.momentum*6.62607004e-34*1e9

    return ray_tracer_force


def make_scene(state, params):
    if params["ray-factory"]["origin"]["type"] == "offset":
        origin = state[:3] + array(params["ray-factory"]["origin"]["value"])
    else:
        origin = params["ray-factory"]["origin"]["value"]
    rf = getattr(ptracer, params["ray-factory"]["type"])(origin=origin, **params["ray-factory"]["params"])
    objects = []
    for i, obj in enumerate(params["objects"]):
        if obj["origin"]["type"] == "offset":
            origin = state[:3] + array(obj["origin"]["value"])
            active = True
        else:
            origin = array(obj["origin"]["value"])
            active = False
        objects.append(getattr(ptracer, obj["type"])(origin, **obj["params"], active=active))
    # print(objects[0].origin)
    # print(t)
    return ptracer.Scene(rf, objects)