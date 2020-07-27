# The _forces_ library.

This folder is a library of Python files, each corresponding
to a force that can be used in the final simulation.

For each Python file _name_.py there must be a _name_.yaml
file, which describes the parameters needed to configure
the force.

Each Python file must contain a function called factory,
which takes in two dictionaries as arguments:
* config, which is the full configuration dictionary for
  the experiment,
* params, which contains the parameters as described in 
  the aforementioned .yaml file.
  
The factory function must return a force function. This
function must take two arguments:
* state, which is a description of the current state of the
  system, in the format [x, y, v_x, v_y]
* t, which is the current time as a float

The force function then returns a 2-element list containing
the x and y components of the force.