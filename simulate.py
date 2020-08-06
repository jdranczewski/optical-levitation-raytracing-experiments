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
    """
    This function returs the derivatives of x, y, v_x, and v_y for a given state and a set of forces.

    :param t: current time
    :param state: [x, y, v_x, v_y]
    :param forces: a list of force functions, as defined in the forces library
    :param mass: mass of the particle in kilograms
    :return: d[x, y, v_x, v_y]/dt
    """
    # print(t)
    acc = np.sum([force(state, t) for force in forces], axis=0)/mass
    # print(state[:2], acc)
    return np.array([state[3], state[4], state[5], acc[0], acc[1], acc[2]])


# This is to allow constructing lists in yaml config files.
# The signature is !linspace start,end,N
def linspace_constructor(loader, node):
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


def main():
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

    fig, ax = plt.subplots()

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
                    # ... and include that value in the variable'd definition in the yaml file
                    text = text.replace("__{}__".format(var["name"]), "{:e}".format(val))
                # Reload the config, with the variable's value in place
                config = yaml.safe_load(text)
        else:
            val = None

        sim_params = config["params"]

        # Construct a list of force functions
        forces = []
        for force in config["forces"]:
            # Import a force by its name as specified in the config
            m = import_module("forces." + force["type"])
            forces.append(m.factory(config, force["params"]))
        if not len(forces):
            raise Exception("No forces were defined.")

        # Define the time axis
        times = np.linspace(sim_params["start"], sim_params["end"], int(sim_params["steps"]))
        # print(sim_params["start"], sim_params["end"], int(sim_params["steps"]))

        # Do the actual calculation
        res = odeint(derivatives, sim_params["initial-conditions"], times, args=(forces, sim_params["mass"]), tfirst=True)

        # Plot the result
        ax.plot(res[:, 0], res[:, 2], "o-", ms=3, label="{:.2e}".format(vs if val is None else val))

    ax.legend()

    # xs = []
    # ys = []
    # fx = []
    # fy = []
    # for x in np.linspace(-2.5e-5, 4.2e-5, 20):
    #     print(x)
    #     for y in np.linspace(600e-6, 1.2*np.amax(res[:,1]), 30):
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
    # ax.axis("equal")
    # for force in config["forces"]:
    #     m = import_module("forces." + force["type"])
    #     if force["type"] == "ray_tracer":
    #         rt_params = force["params"]
    # scene = make_scene(np.array(sim_params["initial-conditions"]), rt_params)
    # scene.run()
    # scene.propagate(50e-6)
    # scene.plot(ax)
    # ax.quiver(xs, ys, fx, fy, zorder=3)

    # z = np.linspace(0, np.amax(res[:,1]), 100)
    # waist_radius = rt_params["ray-factory"]["params"]["waist_radius"]
    # w = waist_radius * np.sqrt(1 + ((z * 600 * 1e-9) / (np.pi * waist_radius ** 2)) ** 2)
    # ax.plot(w, z)

    fig, ax = plt.subplots()
    ax.plot(times, res[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
