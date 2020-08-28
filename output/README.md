# Your output directories go here.

To specify output parameters (including the folder name), modify your simulation's
configuration .yaml file.

## The standard form format the .csv files
is as follows:

* time (s)
* x
* y
* z (m)
* v_x
* v_y
* v_z (m/s)
* qaternion - real part
* qaternion - i part
* qaternion - j part
* qaternion - k part
* omega_x
* omega_y
* omega_z (rad/s, in a target - fixed frame).

See [Wikipedia](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
for an intro on what a quaternion is, or
[this excellent series](https://eater.net/quaternions)
of interactive demonstrations by 3Blue1Brown, which may take a while, but
is great for understanding.
