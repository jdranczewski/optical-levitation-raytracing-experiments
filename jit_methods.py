import numpy as np
from numba import jit


@jit(nopython=True)
def refract(os, dirs, weights, wavelength, normals, n_in, n_out):
    # Establish normals, n1, and n2 arrays based on whether the rays are going in or out
    # cos_i = -np.einsum("ij,ij->i", dirs, normals)
    cos_i = -np.sum(dirs*normals, axis=1)
    going_out = cos_i < 0
    normals[going_out] *= -1
    n1 = np.full(len(os), n_out)
    n2 = np.full(len(os), n_in)
    n1[going_out] = n_in
    n2[going_out] = n_out
    cos_i[going_out] *= -1

    # Calculate a few more angles
    sin_i2 = 1 - cos_i ** 2
    cos_t = np.sqrt(1 - (n1 / n2) ** 2 * (1 - cos_i ** 2))

    # Calculate the initial momentum of the rays
    init_weights = n1 * weights

    # Mark out rays that undergo total internal reflection
    tir = np.sqrt(sin_i2) > n2 / n1
    ntir = np.invert(tir)

    # Calculate the reflection and refraction coefficients
    R_perp = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    R_para = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    R = (R_perp + R_para) / 2
    T = 1 - R

    # Calculate the ray's new direction under refraction and reflection
    # d_refr = (np.einsum("i,ij->ij", n1 / n2, dirs) +
    #           np.einsum("i,ij->ij", (n1 / n2 * cos_i - np.sqrt(1 - (n1 / n2) ** 2 * sin_i2)), normals))
    # d_refl = dirs + 2 * np.einsum("ij,i->ij", normals, cos_i)
    d_refr = ((n1/n2).reshape((-1, 1)) * dirs +
              (n1 / n2 * cos_i - np.sqrt(1 - (n1 / n2) ** 2 * sin_i2)).reshape((-1,1)) * normals)
    d_refl = dirs + 2 * normals * cos_i.reshape((-1, 1))

    # For rays that undergo total internal reflection, assign the reflected direction as the main one
    d_refr[tir] = d_refl[tir]

    # Create new rays for the reflections
    new_d = d_refl[ntir]
    new_weights = weights[ntir] * R[ntir]
    new_origins = os[ntir]

    # Distribute ray weights
    weights[ntir] *= T[ntir]

    # Calculate the change of momentum
    n2[tir] = n1[tir]
    # momentum = (np.einsum("ij,i->j", d_refr, n2 * weights) +
    #             np.einsum("ij,i->j", new_d, n1[ntir] * new_weights) -
    #             np.einsum("ij,i->j", dirs, init_weights)) / wavelength
    momentum = (np.sum(d_refr * (n2*weights).reshape((-1,1)), axis=0) +
                np.sum(new_d * (n1[ntir]*new_weights).reshape((-1,1)), axis=0) -
                np.sum(dirs * init_weights.reshape((-1,1)), axis=0)) / wavelength

    return momentum, d_refr, weights, new_d, new_weights, new_origins


@jit(nopython=True)
def intersect_d_triangles(os, dirs, a, edge1, edge2):
    d = np.full(len(os), np.inf)

    p = np.cross(dirs, edge2)
    # det = np.einsum("j,ij->i", edge1, p)
    det = np.sum(p*edge1, axis=1)
    mask = np.abs(det) > 0
    det[~mask] = 1
    # print(mask)
    # if (~mask).all():
    #     return d

    t = os - a
    # u = np.einsum("ij,ij->i", t, p) / det
    u = np.sum(p*t, axis=1) / det
    mask = mask & (u >= 0) & (u <= 1)
    # print(mask)
    # if (~mask).all():
    #     return d

    q = np.cross(t, edge1)
    # v = np.einsum("ij,ij->i", dirs, q) / det
    v = np.sum(dirs*q, axis=1) / det
    mask = mask & (v >= 0) & (u + v <= 1)
    # print(mask)
    # if (~mask).all():
    #     return d

    # d[mask] = np.einsum("j,ij->i", edge2, q[mask]) / det[mask]
    d[mask] = np.sum(edge2*q[mask], axis=1) / det[mask]
    d[d < 0] = np.inf
    return d