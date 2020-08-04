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

eha = []
ehb = []
ehc = []
ehd = []

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
    fs = [force(state, t) for force in forces]
    acc = np.sum(fs, axis=0)/mass
    # print(state[:2], acc)
    eha.append([state[0], state[1], fs[0][0]/mass, fs[0][1]/mass])
    ehb.append([state[0], state[1], fs[1][0]/mass, fs[1][1]/mass])
    ehc.append([state[0], state[1], fs[2][0]/mass, fs[2][1]/mass])
    ehd.append([state[0], state[1], acc[0], acc[1]])
    return np.array([state[2], state[3], acc[0], acc[1]])


# This is to allow constructing lists in yaml config files.
# The signature is !linspace start,end,N
def linspace_constructor(loader, node):
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


def main():
    global eha, ehb, ehc, ehd
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
        ax.plot(res[:, 0], res[:, 1], "o-", ms=3, label="{:.2e}".format(vs if val is None else val))

    # xs = []
    # ys = []
    # fx = []
    # fy = []
    # for x in np.linspace(-1.5e-6, .3e-6, 20):
    #     print(x)
    #     for y in np.linspace(100e-6, 1.2*np.amax(res[:,1]), 30):
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
    eha = np.array(eha)
    ehb = np.array(ehb)
    ehc = np.array(ehc)
    ehd = np.array(ehd)
    ax.quiver(eha[:,0], eha[:,1], eha[:,2], eha[:,3], color="C1", scale=.5e4, label="Drag")
    ax.quiver(ehb[:, 0], ehb[:, 1], ehb[:, 2], ehb[:, 3], color="C2", scale=.5e4, label="Gravity")
    ax.quiver(ehc[:, 0], ehc[:, 1], ehc[:, 2], ehc[:, 3], color="C3", scale=.5e4, label="Optical")
    ax.quiver(ehd[:, 0], ehd[:, 1], ehd[:, 2], ehd[:, 3], color="C4", scale=.5e4, label="Total", zorder=3)

    # z = np.linspace(0, np.amax(res[:,1]), 100)
    # waist_radius = rt_params["ray-factory"]["params"]["waist_radius"]
    # w = waist_radius * np.sqrt(1 + ((z * 600 * 1e-9) / (np.pi * waist_radius ** 2)) ** 2)
    # ax.plot(w, z)
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(times, res[:,1])
    plt.show()


if __name__ == "__main__":
    main()
