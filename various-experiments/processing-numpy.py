"""
This is a prototype exploring the creation of multiple processes and sharing of an array between them.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments

"""

import numpy as np
import multiprocessing as mp
import ctypes


def f(a, i):
    a[:] = i


if __name__ == "__main__":
    n = 52
    a = mp.Array(ctypes.c_double, n)
    npa = np.frombuffer(a.get_obj())
    npa[:] = np.zeros(n)
    print(npa)
    batch = int(np.ceil(n / 5))
    for i in range(5):
        p = mp.Process(target=f, args=(npa[batch*i:batch*(i+1)], i))
        p.start()
    p.join()
    # f(a, 2)
    print(npa)
