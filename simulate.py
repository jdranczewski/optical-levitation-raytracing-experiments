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
from scipy.integrate import odeint
from importlib import import_module
import matplotlib.pyplot as plt
from forces.ray_tracer import make_scene


def derivatives(t, state, forces, mass):
    # print(t)
    acc = np.sum([force(state, t) for force in forces], axis=0)/mass
    return np.array([state[2], state[3], acc[0], acc[1]])


def linspace_constructor(loader, node):
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    if config["variables"] is not None:
        var_steps = config["variables"]["steps"]
        variables = config["variables"]["vars"]

        fig, ax = plt.subplots()
        for vs in range(var_steps):
            print("var_step")
            with open("config.yaml", 'r') as f:
                text = f.read()
                for var in variables:
                    val = var["start"] + (var["end"]-var["start"])*vs/(var_steps-1)
                    text = text.replace("__{}__".format(var["name"]), str(val))
                config = yaml.safe_load(text)
            sim_params = config["params"]
            forces = []
            for force in config["forces"]:
                m = import_module("forces." + force["type"])
                forces.append(m.factory(config, force["params"]))
            if not len(forces):
                raise Exception("No forces were defined.")

            times = np.linspace(sim_params["start"], sim_params["end"], int(sim_params["steps"]))
            print(sim_params["start"], sim_params["end"], int(sim_params["steps"]))
            res = odeint(derivatives, sim_params["initial-conditions"], times, args=(forces, 1), tfirst=True)
            ax.plot(res[:, 0], res[:, 1], "o-", ms=3, label=vs)
        ax.legend()

        xs = []
        ys = []
        fx = []
        fy = []
        # for x in np.linspace(247e-6, 249.5e-6, 30):
        #     print(x)
        #     for y in np.linspace(-0.4e-6, 0.4e-6, 20):
        #         forces = []
        #         for force in config["forces"]:
        #             m = import_module("forces." + force["type"])
        #             forces.append(m.factory(config, force["params"]))
        #             if force["type"] == "ray_tracer":
        #                 rt_params = force["params"]
        #         if not len(forces):
        #             raise Exception("No forces were defined.")
        #
        #         acc = derivatives(0, np.array((x, y, 0, 0)), forces, sim_params["mass"])
        #
        #         xs.append(x)
        #         ys.append(y)
        #         fx.append(acc[2])
        #         fy.append(acc[3])
        ax.axis("equal")
        for force in config["forces"]:
            m = import_module("forces." + force["type"])
            if force["type"] == "ray_tracer":
                rt_params = force["params"]
        scene = make_scene(np.array(sim_params["initial-conditions"]), rt_params)
        # scene.run()
        # scene.propagate(50e-6)
        scene.plot(ax)
        # ax.quiver(xs, ys, fx, fy, zorder=3)

        plt.show()

    else:
        sim_params = config["params"]
        forces = []
        for force in config["forces"]:
            m = import_module("forces."+force["type"])
            forces.append(m.factory(config, force["params"]))
        if not len(forces):
            raise Exception("No forces were defined.")

        times = np.linspace(sim_params["start"], sim_params["end"], sim_params["steps"])
        res = odeint(derivatives, sim_params["initial-conditions"], times, args=(forces, 1), tfirst=True)
        print(res)
        fig, ax = plt.subplots()
        ax.plot(res[:, 0], res[:, 1], "o-", ms=3)
        plt.show()



if __name__ == "__main__":
    main()
