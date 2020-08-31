"""
This file contains some functions that are either heavy computationally, or called thousands of times.
They have been stripped of all Python fanciness, they take numpy arrays and do numpy things to them.
This is required for compilation with numba (http://numba.pydata.org/), which makes all of this run much faster.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments
"""

import numpy as np
from numba import jit

kwargs = {
    "nopython": True,
    "cache": True,
}


@jit(**kwargs)
def rotate(points, q):
    res = np.zeros((len(points), 4))
    # q*p
    res[:,0] = -np.sum(q[1:]*points, axis=1)
    res[:,1:] = q[0]*points + np.cross(q[1:], points)
    # (q*p) * q^-1
    res2 = np.zeros((len(points), 4))
    res2[:,0] = res[:,0]*q[0] - np.dot(res[:,1:].copy(), -q[1:].copy())
    res2[:,1:] = res[:,0].copy().reshape((-1,1))*(-q[1:]) + (q[0]*res[:,1:]) + np.cross(res[:,1:], -q[1:])
    return res2[:,1:]


@jit(**kwargs)
def refract(os, dirs, weights, wavelength, normals, n_in, n_out, ang_origin):
    # Establish normals, n1, and n2 arrays based on whether the rays are going in or out
    # cos_i = -np.einsum("ij,ij->i", dirs, normals)
    cos_i = -np.sum(dirs*normals, axis=1)
    going_out = cos_i < 0
    normals[going_out] *= -1
    # for each ray n1 is for the material it is currently in, and n2 is for the one it'g going into
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
    # Note the values of n2 are switched below for rays that have undergone tir
    # to make the calculation easier.
    n2[tir] = n1[tir]

    final = d_refr * (n2*weights).reshape((-1,1))
    final[ntir] += new_d * (n1[ntir]*new_weights).reshape((-1, 1))
    final -= dirs * init_weights.reshape((-1, 1))
    momentum = np.sum(final, axis=0) / wavelength
    ang_momentum = np.sum(np.cross((os - ang_origin), final), axis=0)

    return momentum, ang_momentum, d_refr, weights, new_d, new_weights, new_origins


@jit(**kwargs)
def reflect(os, dirs, weights, wavelength, normals, n_in, n_out, ang_origin):
    # cos_i = -np.einsum("ij,ij->i", dirs, normals)
    cos_i = -np.sum(dirs*normals, axis=1)
    going_out = cos_i < 0
    n1 = np.full(len(os), n_out)
    n1[going_out] = n_in

    # Reflect the rays and store the momentum change
    # change = 2 * np.einsum("ij,i->ij", normals, cos_i)
    change = 2 * normals*cos_i.reshape((-1,1))
    forces = change*(n1*weights).reshape((-1, 1))
    momentum = np.sum(forces, axis=0) / wavelength
    ang_momentum = np.sum(np.cross((os-ang_origin), forces), axis=0) / wavelength
    dirs += change

    return momentum, ang_momentum, dirs, weights


@jit(**kwargs)
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


@jit(**kwargs)
def intersect_d_mesh(os, dirs, a, edge1, edge2, margin=1e-2):
    n_rays = len(dirs)
    n_tris = len(a)
    d = np.zeros((n_rays, n_tris))

    p = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        p[:, i, :] = np.cross(dirs, edge2[i])
    # print(p)

    det = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        det[:, i] = np.sum(p[:, i] * edge1[i], axis=1)
    mask = np.abs(det) > 0
    for i in range(n_tris):
        det[:, i][~mask[:, i]] = 1
    # print(len(mask[0]))

    t = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        t[mask[:, i], i, :] = os[mask[:, i]] - a[i]
    # print(t)

    u = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        u[mask[:, i], i] = np.sum(p[mask[:, i], i, :] * t[mask[:,i], i, :], axis=1) / det[mask[:, i], i]
    # print(u)
    mask = mask & (u >= -margin) & (u <= 1+margin)

    q = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        q[mask[:, i], i, :] = np.cross(t[mask[:, i], i, :], edge1[i])

    v = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        v[mask[:, i], i] = np.sum(dirs[mask[:, i]] * q[mask[:, i], i, :], axis=1) / det[mask[:, i], i]
    mask = mask & (v >= -margin) & (u + v <= 1+margin)

    for i in range(n_tris):
        d[mask[:, i], i] = np.sum(edge2[i] * q[mask[:, i], i, :], axis=1) / det[mask[:, i], i]
    for i in range(n_tris):
        d[:, i][d[:, i] < 0] = np.inf
        d[:, i][~mask[:, i]] = np.inf

    m = np.zeros(n_rays)
    for i in range(n_rays):
        m[i] = np.amin(d[i])

    return m, d


@jit(**kwargs)
def mesh_normals(d, n):
    # Mask out the rays with no collisions
    collided = np.count_nonzero(d != np.inf, axis=1) > 0
    d = d[collided]

    ns = np.zeros((d.shape[0], 3))
    for i in range(d.shape[0]):
        m = d[i].min()
        # print(np.abs((d[i] - m)/m))
        indices = np.nonzero((np.abs((d[i] - m)/m) < 1e-5) & (d[i] != np.inf))
        # print((np.abs((d[i] - m)/m)))
        # print(indices)
        for j in range(3):
            ns[i, j] = np.mean(n[indices][:, j])
    # print("-"*15)
    return ns / np.sqrt(ns[:, 0]**2 + ns[:, 1]**2 + ns[:, 2]**2).reshape((-1, 1))


@jit(**kwargs)
def intersect_d_mesh_smooth(os, dirs, a, edge1, edge2, na, nb, nc, margin=0):
    n_rays = len(dirs)
    n_tris = len(a)
    d = np.zeros((n_rays, n_tris))

    p = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        p[:, i, :] = np.cross(dirs, edge2[i])
    # print(p)

    det = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        det[:, i] = np.sum(p[:, i] * edge1[i], axis=1)
    mask = np.abs(det) > 0
    for i in range(n_tris):
        det[:, i][~mask[:, i]] = 1
    # print(len(mask[0]))

    t = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        t[mask[:, i], i, :] = os[mask[:, i]] - a[i]
    # print(t)

    u = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        u[mask[:, i], i] = np.sum(p[mask[:, i], i, :] * t[mask[:,i], i, :], axis=1) / det[mask[:, i], i]
    # print(u)
    mask = mask & (u >= -margin) & (u <= 1+margin)

    q = np.zeros((n_rays, n_tris, 3))
    for i in range(n_tris):
        q[mask[:, i], i, :] = np.cross(t[mask[:, i], i, :], edge1[i])

    v = np.zeros((n_rays, n_tris))
    for i in range(n_tris):
        v[mask[:, i], i] = np.sum(dirs[mask[:, i]] * q[mask[:, i], i, :], axis=1) / det[mask[:, i], i]
    mask = mask & (v >= -margin) & (u + v <= 1+margin)

    for i in range(n_tris):
        d[mask[:, i], i] = np.sum(edge2[i] * q[mask[:, i], i, :], axis=1) / det[mask[:, i], i]
    for i in range(n_tris):
        d[:, i][d[:, i] < 0] = np.inf
        d[:, i][~mask[:, i]] = np.inf
    # print(d)

    m = np.zeros(n_rays)
    index = np.zeros(n_rays)
    for i in range(n_rays):
        m[i] = np.amin(d[i])
        index[i] = np.argmin(d[i])

    # Compute normals
    w = 1 - u - v
    collided = np.count_nonzero(d != np.inf, axis=1) > 0
    im = index[collided].astype(np.int64)
    ir = np.arange(n_rays)[collided].astype(np.int64)
    normals = np.zeros((len(im), 3))
    for i in range(len(im)):
        # print(na[im[i]])
        # print(w[i, im[i]])
        i_ray = ir[i]
        i_tri = im[i]
        normals[i, :] = na[i_tri]*w[i_ray, i_tri] + nb[i_tri]*u[i_ray, i_tri] + nc[i_tri]*v[i_ray, i_tri]
    # print(normals)

    return m, normals
