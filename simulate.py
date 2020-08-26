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
# Parsing YAML config files
import yaml

# Numerical manipulation and integration
import numpy as np
from scipy.integrate import odeint, solve_ivp

# Display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utilities
from importlib import import_module
from datetime import datetime
from time import sleep
import os
import argparse


def quaternion(angle, vector):
    vector = np.array(vector).astype(float)
    vector /= np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    return np.array((np.cos(angle/2), *np.sin(angle/2)*np.array(vector)))


def quat_prod(p, q):
    # print(p, q)
    res = np.zeros(4)
    p, q = np.array((p,q))
    res[0] = p[0]*q[0] - np.dot(p[1:], q[1:])
    res[1:] = p[0]*q[1:] + q[0]*p[1:] + np.cross(p[1:], q[1:])
    return res


def derivatives(t, state, forces, mass, inertia, pbar):
    """
    This function returs the derivatives of x, y, v_x, and v_y for a given state and a set of forces.

    :param t: current time
    :param state: [x, y, z, v_x, v_y, v_z, q1, q2, q3, q4, w1, w2, w3]
    :param forces: a list of force functions, as defined in the forces library
    :param mass: mass of the particle in kilograms
    :param inertia: a moment of inertia vector in the format np.array(I_x, I_y, I_z) along the object's principal axes
    :param pbar: a tqdm progress bar object
    :return: np.array
    """
    # Update the progress bar
    if t < pbar.total:
        pbar.n = t
        pbar.refresh()
    # Compute the values of the various forces and sum them
    acc = np.sum([force(state, t) for force in forces], axis=0)/mass

    q, w = state[6:10], state[10:]
    moments = quat_prod(quat_prod(q * [1, -1, -1, -1], (0, *acc[3:])), q)[1:]
    # print(acc)

    # Return an array of form [v_x, v_y, v_z, a_x, a_y, a_z]
    return np.concatenate(([state[3], state[4], state[5], acc[0], acc[1], acc[2]],
                           quat_prod(.5 * q, (0, *w)),
                           (moments - np.cross(w, inertia * w)) / inertia))


def linspace_constructor(loader, node):
    """
    This function is a linspace constructor integrated with the YAML parser. Useful for things like
    BasicRF, which takes lists of values as arguments.

    The signature in YAML is  !linspace start,end,N
    """
    value = loader.construct_scalar(node).split(",")
    return np.linspace(float(value[0]), float(value[1]), int(value[2]))


# Add the above function to the YAML parser
yaml.SafeLoader.add_constructor("!linspace", linspace_constructor)


########################################
#     Main logic of the simulation     #
########################################
def main():
    """
    Runs the simulation.

    :return: None
    """
    # We may overwrite this function if there is no forces, so this is to prevent Python from thinking
    # that it's a global variable
    global derivatives

    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs='?', help="the configuration YAML file to use", default="config.yaml")
    args = parser.parse_args()
    config_file = args.config

    tqdm.write("Loading configuration...")

    # Load the config file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Check if any variables have been configured
    if config["variables"] is not None:
        var_steps = config["variables"]["steps"]
        variables = config["variables"]["vars"]
    else:
        var_steps = 1
        variables = []

    # This list will store the res arrays for various variable steps
    final = []

    # Go through variable steps, storing the values in var_values (to be used in the output procedures)
    var_values = [[] for i in range(len(variables))]
    for vs in range(var_steps):
        # If variables have been defined, calculate their current value and push that into the config
        if len(variables):
            with open(config_file, 'r') as f:
                # Read the config file
                text = f.read()
            # For each variable calculate the current value...
            for i, var in enumerate(variables):
                if var_steps > 1:
                    val = var["start"] + (var["end"]-var["start"])*vs/(var_steps-1)
                else:
                    val = var["start"]
                var_values[i].append(val)
                # ... and include that value in the variable'd definition in the yaml file
                text = text.replace("__{}__".format(var["name"]), "{:e}".format(val))

            # Reload the config, with the variable's value in place
            config = yaml.safe_load(text)

        sim_params = config["params"]

        # Construct a list of force functions
        forces = []
        # If there are no forces, replace the derivatives function with a new one
        # (this is done so that the normal derivatives function doesn't have to check whether any forces exist
        # on each run, this would be sad performance-wise)
        if config["forces"] is None:
            raise NotImplementedError
            # derivatives = lambda t, state, forces, mass: np.array([state[3], state[4], state[5], 0, 0, 0])
        else:
            for force in config["forces"]:
                # Import a force factory by its name as specified in the config file
                m = import_module("forces." + force["type"])
                # Let the imported factory make a force function and add that to the list of forces
                forces.append(m.factory(config, force["params"]))

        # Define the time axis
        times = np.linspace(sim_params["start"], sim_params["end"], int(sim_params["steps"]))

        # Set up a progress bar
        pbar = tqdm(total=sim_params["end"])

        # Do the calculation
        kw = sim_params["solve_ivp-kwargs"]
        kw = {} if kw is None else kw
        # res, info = odeint(derivatives, sim_params["initial-conditions"], times,
        #                    args=(forces, sim_params["mass"], pbar), tfirst=True, full_output=True,
        #                    **kw)
        # print(sum(list(info["nfe"])))
        # print(info)
        init = np.concatenate((sim_params["initial-conditions"], quaternion(sim_params["rot"], sim_params["axis"]), sim_params["w-init"]))
        # print(quaternion(sim_params["rot"], sim_params["axis"]))
        res = solve_ivp(derivatives, (sim_params["start"], sim_params["end"]), init,
                        args=(forces, sim_params["mass"], sim_params["inertia"], pbar), t_eval=times, **kw)
        res = res.y.T
        pbar.close()
        # A silly solution to let tqdm clean up before upcoming prints
        sleep(0.1)

        # Save the result in the 'final' list
        final.append(np.column_stack((times, res)))

    # Output
    tqdm.write("Making the output...")

    out_params = config["output"]
    labels = ("time", "x", "y", "z", "v_x", "v_y", "v_z")
    # Labels on the 'main' graph are in a different order
    labels_main = ("z", "x", "y")
    var_names = [var["name"] for var in variables]

    # Determines the path under which output will be saved (if the user wants saving)
    if out_params["save-as"] is not None:
        # We allow time tags in the directory name, as defined in https://strftime.org/
        now = datetime.now()
        name = now.strftime(out_params["save-as"])
        savepath = os.path.join("output", name)
        # Make the directory
        os.mkdir(savepath)
        tqdm.write("The output will be saved in " + savepath)

        # Copy the config file to the directory
        with open(config_file, 'r') as f:
            with open(os.path.join(savepath, "config.yaml"), 'w') as fw:
                fw.write(f.read())

        # If desired by the user, we save the data produced as csv files
        # The 'sparse' parameter lets us save every n rows
        if out_params["csv"]["save"]:
            for i, f in enumerate(final):
                # Variable values for that iteration are saved in the header
                header = ", ".join(["{}: {:e}".format(var_names[j], var_values[j][i]) for j in range(len(var_names))])
                np.savetxt(os.path.join(savepath, str(i)+".csv"), f[::out_params["csv"]["sparse"]], delimiter=",", header=header)
    else:
        savepath = ""

    # Plot the "main" graph - a 3D view and time graphs for each axis
    if out_params["show-main-graph"] or len(savepath):
        fig = plt.figure(figsize=(12.0, 10.0))
        ax = [fig.add_subplot(221, projection='3d')] + [fig.add_subplot(2,2,i) for i in range(2,5)]

        for i, f in enumerate(final):
            if len(var_names):
                legend = ", ".join(["{}: {:.2e}".format(var_names[j], var_values[j][i]) for j in range(len(var_names))])
            else:
                legend = ""
            ax[0].plot(f[:, 1], f[:, 2], f[:, 3])
            ax[1].plot(f[:, 0], f[:, 3])
            ax[2].plot(f[:, 0], f[:, 1])
            ax[3].plot(f[:, 0], f[:, 2], label=legend)
            ax[0].plot([f[0, 1]], [f[0, 2]], [f[0, 3]], "o", c="tab:grey")

        # Cosmetics
        for i, axis in enumerate(ax[1:]):
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("{} (m)".format(labels_main[i]))
            axis.grid()
        if len(var_names):
            ax[3].legend()
        fig.tight_layout()

        # Save the graph
        if len(savepath):
            fig.savefig(os.path.join(savepath, "main.png"), bbox_inches="tight")
        # If the user doesn't want to show the graph, close it here
        if not out_params["show-main-graph"]:
            plt.close(fig)

    # Render, save, and display the specified custom graphs
    if out_params["graphs"] is not None:
        for graph in out_params["graphs"]:
            fig, ax = plt.subplots()

            for i, f in enumerate(final):
                if len(var_names):
                    legend = ", ".join(["{}: {:.2e}".format(var_names[j], var_values[j][i]) for j in range(len(var_names))])
                else:
                    legend=""
                ax.plot(f[:, graph["x"]["column"]]/graph["x"]["scale"],
                        f[:, graph["y"]["column"]]/graph["y"]["scale"],
                        label=legend)

            # Cosmetics
            ax.set_xlabel(r"{} ({})".format(
                labels[graph["x"]["column"]] if graph["x"]["label"] is None else graph["x"]["label"],
                graph["x"]["unit"]
            ))
            ax.set_ylabel(r"{} ({})".format(
                labels[graph["y"]["column"]] if graph["y"]["label"] is None else graph["y"]["label"],
                graph["y"]["unit"]
            ))
            ax.grid()
            if len(var_names):
                ax.legend()

            # Save and close the graph, depending on config
            if len(savepath) and graph["name"] is not None:
                fig.savefig(os.path.join(savepath, graph["name"]+".png"), bbox_inches="tight")
            if not graph["show"]:
                plt.close(fig)

    # Show all the graphs that haven't been closed
    plt.show()


if __name__ == "__main__":
    main()
