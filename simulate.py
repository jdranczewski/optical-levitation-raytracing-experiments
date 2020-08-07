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
from tqdm import tqdm
from forces.ray_tracer import make_scene


def derivatives(t, state, forces, mass):
    """
    This function returs the derivatives of x, y, v_x, and v_y for a given state and a set of forces.

    :param t: current time
    :param state: [x, y, v_x, v_y]
    :param forces: a list of force functions, as defined in the forces library
    :param mass: mass of the particle in kilograms
    :return: d[x, y, v_x, v_y]/dt
    """
    # print(t)
    forces_v = [force(state, t) for force in forces]
    acc = np.sum(forces_v, axis=0)/mass
    # print("-" * 20)
    # print(state[:3])
    # for i in range(len(forces)):
    #     print(forces[i].__name__, forces_v[i]/mass)
    return np.array([state[3], state[4], state[5], acc[0], acc[1], acc[2]])


# This is to allow constructing lists in yaml config files.
# The signature is !linspace start,end,N
def linspace_constructor(loader, node):
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


def main():
    global derivatives
    """
    Runs the simulation.

    :return: None
    """
    # Load the config file
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Check if any variables have been configured
    if config["variables"] is not None:
        var_steps = config["variables"]["steps"]
        variables = config["variables"]["vars"]
    else:
        var_steps = 1
        variables = []

    final = []

    # Go through variable steps
    for vs in range(var_steps):
        print("var_step")
        # If variables have been defined, calculate their current value and push that into the config
        if len(variables):
            with open("config.yaml", 'r') as f:
                # Read the config file
                text = f.read()
                # For each variable calculate the current value...
                for var in variables:
                    if var_steps > 1:
                        val = var["start"] + (var["end"]-var["start"])*vs/(var_steps-1)
                    else:
                        val = var["start"]
                    print(val)
                    # ... and include that value in the variable'd definition in the yaml file
                    text = text.replace("__{}__".format(var["name"]), "{:e}".format(val))
                # Reload the config, with the variable's value in place
                config = yaml.safe_load(text)
        else:
            val = None

        sim_params = config["params"]

        # Construct a list of force functions
        forces = []
        if config["forces"] is None:
            derivatives = lambda t, state, forces, mass: np.array([state[3], state[4], state[5], 0, 0, 0])
        else:
            for force in config["forces"]:
                # Import a force by its name as specified in the config
                m = import_module("forces." + force["type"])
                forces.append(m.factory(config, force["params"]))

        # Define the time axis
        times = np.linspace(sim_params["start"], sim_params["end"], int(sim_params["steps"]))
        # print(sim_params["start"], sim_params["end"], int(sim_params["steps"]))

        # Do the actual calculation
        steps = sim_params["progress-check"]["steps"]
        tstep = int(sim_params["steps"]) // steps
        res = np.zeros((sim_params["steps"], 6))
        init = np.array(sim_params["initial-conditions"])
        for i in tqdm(range(steps)):
            start, end = i*tstep, (i+1)*tstep
            if i+1 == steps:
                end = len(times)
            res[start:end] = odeint(derivatives, init, times[start:end], args=(forces, sim_params["mass"]), tfirst=True)
            init = res[end - 1]

        # Store the result
        final.append(np.column_stack((res, times)))

    # print(final)
    fig = plt.figure()
    ax = [fig.add_subplot(221, projection='3d')] + [fig.add_subplot(2,2,i) for i in range(2,5)]
    for f in final:
        ax[0].plot(f[:, 0], f[:, 1], f[:, 2])
        ax[1].plot(f[:, 6], f[:, 2])
        ax[2].plot(f[:, 6], f[:, 0])
        ax[3].plot(f[:, 6], f[:, 1])
    l = ("x", "y", "z")
    for i, axis in enumerate(ax[1:]):
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("{} (m)".format(l[i]))

    # z = np.linspace(0,-0.006,10)
    # w = 3.39728e-6 * np.sqrt(1 + ((z * 532 * 1e-9) / (np.pi * 3.39728e-6 ** 2)) ** 2)
    # ax[0].plot(w, np.zeros_like(w), z)
    # ax[0].plot(-w, np.zeros_like(w), z)
    # ax[0].plot(np.zeros_like(w), w, z)
    # ax[0].plot(np.zeros_like(w), -w, z)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    for f in final:
        ax.plot(f[:, 0], f[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
