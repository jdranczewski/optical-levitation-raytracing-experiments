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


def derivatives(t, state, forces, mass):
    acc = np.sum([force(state, t) for force in forces], axis=0)/mass
    return np.array([state[2], state[3], acc[0], acc[1]])


def main():
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    if config["variables"] is not None:
        var_steps = config["variables"]["steps"]
        variables = config["variables"]["vars"]

        for vs in range(var_steps):
            with open("config.yaml", 'r') as f:
                text = f.read()
                for var in variables:
                    val = var["start"] + (var["end"]-var["start"])*vs/var_steps
                    text = text.replace("__{}__".format(var["name"]), str(val))
                    print(var["name"])
                config = yaml.safe_load(text)
            sim_params = config["params"]
            forces = []
            for force in config["forces"]:
                m = import_module("forces." + force["type"])
                forces.append(m.factory(config, force["params"]))
            if not len(forces):
                raise Exception("No forces were defined.")

            times = np.linspace(sim_params["start"], sim_params["end"], int(sim_params["steps"]))
            res = odeint(derivatives, sim_params["initial-conditions"], times, args=(forces, 1), tfirst=True)
            fig, ax = plt.subplots()
            ax.plot(res[:, 0], res[:, 1], "o-", ms=3)
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
        fig, ax = plt.subplots()
        ax.plot(res[:, 0], res[:, 1], "o-", ms=3)
        plt.show()



if __name__ == "__main__":
    main()
