"""
This is a library for simulating 2D motion of a target under a set ot forces, most importantly a ray-tracing-based
force calculator for arbitrary objects in laser beams.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments

NOTE: All momenta values need to be multiplied by h (Planck's constant) * 1e9 (wavelength is stored in nm)
"""
import yaml
import numpy as np
from forces.ray_tracer import make_scene
from importlib import import_module
import matplotlib.pyplot as plt


def derivatives(t, state, forces, mass):
    # print(t)
    acc = np.sum([force(state, t) for force in forces], axis=0)/mass
    return np.array([state[2], state[3], acc[0], acc[1]])


def linspace_constructor(loader, node):
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


def main():
    with open("quiver_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        sim_params = config["params"]

    xs = []
    ys = []
    fx = []
    fy = []
    for x in np.linspace(0, 120e-6, 30):
        print(x)
        for y in np.linspace(-20e-6, 20e-6, 20):
            forces = []
            for force in config["forces"]:
                m = import_module("forces." + force["type"])
                forces.append(m.factory(config, force["params"]))
                if force["type"] == "ray_tracer":
                    rt_params = force["params"]
            if not len(forces):
                raise Exception("No forces were defined.")

            acc = derivatives(0, np.array((x, y, 0, 0)), forces, sim_params["mass"])

            xs.append(x)
            ys.append(y)
            fx.append(acc[2])
            fy.append(acc[3])
    fig, ax = plt.subplots()
    ax.axis("equal")
    scene = make_scene(np.array((x, y, 0, 0)), rt_params)
    scene.run()
    scene.propagate(150e-6)
    scene.plot(ax)
    ax.quiver(xs, ys, fx, fy, zorder=3)
    plt.show()


if __name__ == "__main__":
    main()
