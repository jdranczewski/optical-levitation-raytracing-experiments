# The _forces_ library.

This folder is a library of Python files, each corresponding
to a force that can be used in the final simulation.

Each Python file specifies the needed config parameters in a 
comment line close to the top of the file.

Each Python file must contain a function called factory,
which takes in two dictionaries as arguments:
* config, which is the full configuration dictionary for
  the experiment,
* params, which contains the force parameters as described in 
  the aforementioned .yaml file.
  
The factory function must return a force function. This
function must take two arguments:
* state, which is a description of the current state of the
  system, in the format
  [x, y, z, v_x, v_y, v_z, q_r, q_i, q_j, q_k, w1, w2, w3]
* t, which is the current time as a float

The force function then returns a 6-element list containing
the x, y, and z components of the force, as well as the torques in
the x, y, and z directions. The returned values should be in SI units.