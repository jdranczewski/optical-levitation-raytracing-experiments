"""
3D ray tracing library accounting for momentum changes

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

There should be a Jupyter Notebook in the `various-experiments` directory called "ptracer-experiments.ipynb" - this
contains examples of most things that can be done with this library.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments

NOTE: All momenta values need to be multiplied by h (Planck's constant) * 1e9 (wavelength is stored in nm)
"""


import numpy as np
import jit_methods as jm
from random import random
from multiprocessing import Process, Queue


############################
#     Helper functions     #
############################
def normalize(x):
    """
    Given any vector, return a vector of unit length pointing in the same direction.

    :param x: a vector in np.array form
    :return: np.array - normalised vector
    """
    x = np.array(x)
    return x / np.sqrt(x.dot(x))


def normalize_array(x):
    """
    Given an array of vectors, return an array of vectors of unit length pointing in the same direction.

    :param x: an array of vectors in np.array form
    :return: np.array - an array of normalised vectors
    """
    x = x.astype(float)
    norm = np.sqrt(np.einsum('...i,...i', x, x))
    mask = norm > 0
    norm = norm.reshape(-1, 1)
    x[mask, :] = x[mask, :] / norm[mask]
    return x


def rot_to_vector(points, vector):
    """
    Rotate a set of points initially on the (x,y) plane so that the new normal of their plane is the given vector.

    :param points: np.array of points, [[X,Y,Z], [X,Y,Z], ...]
    :param vector: 3d normalised vector as a np.array, [X,Y,Z]
    :return:
    """
    # Rotate around y
    c = vector[2]
    s = np.sqrt(1 - c ** 2)
    rot = np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])
    points = np.dot(points, rot.T)

    # Rotate around z
    angle = np.arctan2(vector[1], vector[0])
    c = np.cos(angle)
    s = np.sin(angle)
    rot = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    return np.dot(points, rot.T)


#################################
#     Main abstract classes     #
#################################
class Scene:
    """
    A container for a ray tracing situation.

    :property self.history: a series of snapshots of ray origins at each step.
    :property self.objects: a list of all the TracerObjects in the scene
    """

    def __init__(self, rf, objects):
        """
        :param rf: a RayFactory object
        :param objects: a list of TracerObject objects
        """
        self.r_origins = rf.origins.copy()
        self.history = [self.r_origins.copy()]
        self.r_dirs = rf.dirs.copy()
        self.r_weights = rf.weights.copy()
        self.r_wavelength = rf.wavelength
        self.active = np.ones(len(self.r_origins)).astype(bool)
        self.objects = objects
        offset = 0
        for i in range(len(objects)):
            if isinstance(self.objects[i+offset], ObjectContainer):
                self.objects.extend(self.objects[i+offset].objects)
                del self.objects[i+offset]
                offset -= 1

    def step(self):
        """
        Make a single ray tracing step - each ray moves towards its next collision.

        :return: None
        """
        # Determine nearest intersection
        d = np.full((len(self.objects), len(self.r_origins)), np.inf)
        for i, obj in enumerate(self.objects):
            d[i][self.active] = obj.intersect_d(self.r_origins[self.active], self.r_dirs[self.active])

        # For each ray, find the index of the object that it should collide with
        collisions = np.argmin(d, axis=0)
        no_collisions = np.count_nonzero(d == np.inf, axis=0) == len(self.objects)
        collisions[no_collisions] = -1
        self.active = no_collisions == False
        # print(self.active)

        # Achieve points of intersection
        self.r_origins[self.active] += np.einsum('ij,i->ij', self.r_dirs[self.active],
                                                 np.amin(d.T[self.active], axis=1))

        # Make empty arrays for new objects
        new_dirs = np.array([])
        new_weights = np.array([])
        new_origins = np.array([])

        # Let objects do things to rays (refraction, reflection, data logging, ...)
        for i, obj in enumerate(self.objects):
            # Get a mask for rays that are colliding with this object
            collided = collisions == i
            self.r_dirs[collided], self.r_weights[collided], nd, nw, no = \
                obj.act_rays(self.r_origins[collided], self.r_dirs[collided],
                             self.r_weights[collided], self.r_wavelength)
            # If new rays have been returned, store their details
            if len(new_dirs) and len(nd):
                new_dirs = np.concatenate((new_dirs, nd))
                new_weights = np.concatenate((new_weights, nw))
                new_origins = np.concatenate((new_origins, no))
            elif len(nd):
                new_dirs = nd
                new_weights = nw
                new_origins = no

        # Handle any new rays
        if len(new_dirs):
            self.r_dirs = np.concatenate((self.r_dirs, new_dirs))
            self.r_weights = np.concatenate((self.r_weights, new_weights))
            self.r_origins = np.concatenate((self.r_origins, new_origins))
            self.active = np.concatenate((self.active, np.ones(len(new_dirs)).astype(bool)))
        self.history.append(self.r_origins.copy())

        # Turn off rays that carry a very low fraction of the total power
        self.active[self.r_weights/np.amax(self.r_weights) < 1e-5] = False

    def propagate(self, d):
        """
        Propagate the rays along their current direction. Doesn't take collisions into account.
        Store the new positions in self.history.

        :param d: distance to propagate
        :return: None
        """
        self.r_origins += self.r_dirs*d
        self.history.append(self.r_origins.copy())

    def run(self, limit=100, queue=None, margin=1e-10, announce_steps=False):
        """
        Run a full ray tracing simulation. Stops when all rays terminated or limit of steps reached.

        :param limit: maximum number of steps
        :param margin: distance to propagate rays after each collision
        :param announce_steps: if True, print step info during each step
        :return: None
        """
        for i in range(limit):
            self.step()
            self.r_origins += self.r_dirs * margin
            if announce_steps:
                print("Step", i)
            if np.all(np.invert(self.active)):
                break
        if queue is not None:
            queue.put(self.momentum)

    def plot(self, ax, show_weight=True, ray_kwargs=None, m_quiver=False, m_quiver_kwargs=None, sparse=1):
        """
        Given a matplotlib axis object, plot all simulation elements onto it.

        :param ax: a matplotlib axis
        :param ray_kwargs: keyword arguments to pass to ax.plot when drawing rays
        :param m_quiver: if True, the changes of momentum are plotted for all TracerObjects in the scene
        :param m_quiver_kwargs: keyword arguments to pass to ax.quiver when drawing the momentum changes, most useful
                                example being "scale", which changes the size of the arrows
        :param sparse: integer, if specified to be n, only every nth ray will be drawn
                       NOTE: sparse rendering may produce unexpected results (like disjoint rays) when
                       ray-splitting occurs in the scene.
        :return: None
        """
        # Not every ray has existed throughout the entire run, hence the great list comprehension below, which
        # constructs paths for all the rays.
        max_w = np.amax(self.r_weights)
        ray_kwargs = {} if ray_kwargs is None else ray_kwargs
        m_quiver_kwargs = {} if m_quiver_kwargs is None else m_quiver_kwargs
        for i, ray_hist in enumerate(
                [[self.history[j][i] for j in range(len(self.history)) if i < len(self.history[j])] for i in
                 range(0, len(self.history[-1]), sparse)]):
            rh = np.array(ray_hist)
            if show_weight:
                ax.plot(rh[:, 0], rh[:, 1], rh[:, 2], alpha=self.r_weights[i]/max_w, **ray_kwargs)
            else:
                ax.plot(rh[:, 0], rh[:, 1], rh[:, 2], alpha=1, **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)
        if m_quiver:
            ms = np.array([obj.momentum for obj in self.objects])
            os = np.array([obj.origin for obj in self.objects])
            # print(os)
            # print(ms)
            mmax = np.sqrt(np.amax(np.sum(ms**2, axis=0)))
            ax.quiver(os[:, 0], os[:, 1], os[:, 2], ms[:, 0], ms[:, 1], ms[:, 2], length=1e-6/mmax, **m_quiver_kwargs)

    @property
    def momentum(self):
        """
        Total momentum acquired by objects in the scene.

        :return: np.array, [X,Y]
        """
        active = [obj.momentum for obj in self.objects if obj.active]
        if len(active):
            return np.einsum("ij->j", np.array(active))
        else:
            return np.array([0, 0, 0])

    @property
    def ang_momentum(self):
        """
        Total momentum acquired by objects in the scene.

        :return: np.array, [X,Y]
        """
        active = [obj.ang_momentum for obj in self.objects if obj.active]
        if len(active):
            return np.einsum("ij->j", np.array(active))
        else:
            return np.array([0, 0, 0])


class MultiScene:
    def __init__(self, rf, obj, n_threads=5):
        self.scenes = []
        batch = int(np.ceil(len(rf.origins) / n_threads))
        self.momentum = np.zeros(3)
        for i in range(n_threads):
            rf2 = RayFactory()
            s = slice(batch*i, batch*(i+1))
            rf2.origins = rf.origins[s]
            rf2.dirs = rf.dirs[s]
            rf2.weights = rf.weights[s]
            rf2.wavelength = rf.wavelength
            self.scenes.append(Scene(rf2, obj))

    def run(self, limit=100, margin=1e-10):
        q = Queue()
        ps = []
        for scene in self.scenes:
            p = Process(target=scene.run, args=(limit, q))
            p.start()
            ps.append(p)
        [p.join() for p in ps]

        for i in range(len(self.scenes)):
            self.momentum += q.get()


class TracerObject:
    """
    Base class for all objects that interact with rays in this ray tracer.

    :parameter self.origin: the object's origin
    :parameter self.momentum: total momentum accumulated by the object
    """
    def __init__(self, origin, n_out=1., n_in=1., ang_origin=None, rot=(1,0,0,0), reflective=False, active=True):
        """
        Create a new TracerObject.

        NOTE about refractive indices: the Rays *do not* keep track of the refractive index of the medium.
        The refraction angle is calculated completely locally, on the assumption that all "volumes" are correctly
        constrained with TracerObjects that have correctly set n_in and n_out.

        This means that when constructing a flat glass slab, you have to make sure that the Surface objects
        constraining it have the same n_in, otherwise you *will* get unphysical behaviour.

        :param origin: Coordinates of the object's centre - [X, Y]
        :param n_out: Refractive index outside of the object
        :param n_in: Refractive index inside of the object
        :param reflective:
        """
        self.origin = np.array(origin).astype(float)
        self.ang_origin = self.origin if ang_origin is None else np.array(ang_origin).astype(float)
        self._rot = np.array(rot).astype(float)
        offset = jm.rotate(np.array([self.origin - self.ang_origin]), self._rot)[0]
        self.origin = self.ang_origin + offset
        self.n_out = float(n_out)
        self.n_in = float(n_in)
        self.momentum = np.zeros(3)
        self.ang_momentum = np.zeros(3)
        self.active = active
        # The object's ray-acting function is assigned to the outwards-facing, general "act_rays" function,
        # unless reflective is None, in which case act_rays is left as is.
        if reflective:
            self.act_rays = self.reflect
        elif reflective is False:
            self.act_rays = self.refract

    def intersect_d(self, os, dirs):
        """
        Given ray data, return the distances the rays have to travel before intersecting this object.

        :param os: np.array of ray origins, [[X,Y], [X,Y], ...]
        :param dirs: np.array of ray directions as unit vectors, [[X,Y], [X,Y], ...]
        :return: np.array of distances for each ray
        """
        raise NotImplementedError

    def normals(self, points):
        """
        Calculate normals to the object.

        :param points: np.array of points at which we wish to know the normals, [[X,Y], [X,Y], ...]
        :return: np.array of unit normals at the given points, [[X,Y], [X,Y], ...]
        """
        raise NotImplementedError

    def act_ray(self, os, dirs, weights, wavelength):
        """
        Call this function when you want rays to interact with this object. It should be overwritten in the __init__
        with either self.reflect or self.refract, unless a child class has a particular implementation.

        :param os: np.array of ray origins, [[X,Y], [X,Y], ...]
        :param dirs: np.array of ray directions as unit vectors, [[X,Y], [X,Y], ...]
        :param weights: np.array of ray weights
        :param wavelength: The ray wavelength in nanometers.
        :return: refracted ray directions, refracted ray weights, new reflected ray: directions, weights, origins
        """
        empty = np.array([])
        return dirs, weights, empty, empty, empty

    def reflect(self, os, dirs, weights, wavelength):
        """
        Calculate reflections for a given set of ray data.

        :param os: np.array of ray origins, [[X,Y], [X,Y], ...]
        :param dirs: np.array of ray directions as unit vectors, [[X,Y], [X,Y], ...]
        :param weights: np.array of ray weights
        :param wavelength: The ray wavelength in nanometers.
        :return: reflected ray directions, weights, 3 * empty np.array, as no new rays are created (this matches
                 the signature of self.act_rays).
        """
        # Compute the normals
        normals = self.normals(os)

        # Determine which refractive index to use for momentum change calculation
        momentum, ang_momentum, dirs, weights = jm.reflect(os, dirs, weights, wavelength, normals, self.n_in, self.n_out, self.ang_origin)
        self.momentum -= momentum
        self.ang_momentum -= ang_momentum

        # Return three empty arrays, as no new rays are created during a reflection
        empty = np.array([])
        return dirs, weights, empty, empty, empty

    def refract(self, os, dirs, weights, wavelength):
        """
        Calculate refraction for a given set of ray data.

        :param os: np.array of ray origins, [[X,Y], [X,Y], ...]
        :param dirs: np.array of ray directions as unit vectors, [[X,Y], [X,Y], ...]
        :param weights: np.array of ray weights
        :param wavelength: The ray wavelength in nanometers.
        :return: refracted ray directions, refracted ray weights, new reflected ray: directions, weights, origins
        """
        # Calculate the normals
        normals = self.normals(os)

        momentum, ang_momentum, d_refr, weights,\
        new_d, new_weights, new_origins = jm.refract(os, dirs, weights, wavelength,
                                                     normals, self.n_in, self.n_out,
                                                     self.ang_origin)
        self.momentum -= momentum
        self.ang_momentum -= ang_momentum

        return d_refr, weights, new_d, new_weights, new_origins

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis.

        :param ax: a matplotlib axis object
        :return: None
        """
        pass


class ObjectContainer:
    def __init__(self, objects):
        self.objects = objects


#################################
#     Various Ray Factories     #
#################################
class RayFactory:
    """
    A container object for ray data. Exposes:

    self.origins: ray origins, np.array, [[X,Y,Z], [X,Y,Z], ...]
    self.dirs: ray directions (normalised), np.array, [[X,Y,Z], [X,Y,Z], ...]
    self.weights: ray weights, np.array, [w, w, ...]
    self.wavelength: ray wavelength, float
    """
    def __init__(self):
        pass

    def __getitem__(self, item):
        new = RayFactory()
        new.origins = self.origins[item]
        new.dirs = self.dirs[item]
        new.weights = self.weights[item]
        new.wavelength = self.wavelength
        return new

    def __add__(self, other):
        rf = RayFactory()
        rf.origins = np.concatenate((self.origins, other.origins))
        rf.dirs = np.concatenate((self.dirs, other.dirs))
        rf.weights = np.concatenate((self.weights, other.weights))
        rf.wavelength = self.wavelength
        if rf.wavelength != other.wavelength:
            raise Exception("Only RayFactories with the same wavelength can be added!")
        return rf


class BasicRF(RayFactory):
    def __init__(self, x, y, z, dir, weight=1, wavelength=600):
        """
        Creates a basic RayFactory. The wavelength is shared between all rays, but the other parameters can be
        supplied either as a single value or as a list of values. If a parameter is given as a single value (say x=2,
        z = 0), but another parameter is given as a list (like y=[0,1,3]), the single value is applied to all the rays,
        giving self.origins = [[2,0,0], [2,1,0], [2,3,0]].

        :param x: a single value (float) or a list of values for the ray origin x position
        :param y: a single value (float) or a list of values for the ray origin y position
        :param z: a single value (float) or a list of values for the ray origin z position
        :param dir: a single vector or a list of vectors for ray directions. Normalised internally
        :param weight: a single value (float) or a list of values for the ray weight
        :param wavelength: a single float for the ray's wavelength
        """
        super().__init__()
        # Store the wavelength
        self.wavelength = float(wavelength)

        # Determine which parameters were given as lists, and their lengths
        n = np.zeros(5)
        for i, arg in enumerate((x, y, z, dir, weight)):
            try:
                n[i] = len(arg)
                # Ensure dir is a list of lists, not just [X,Y,Z] (which would return len=3)
                if arg is dir:
                    len(dir[0])
            except TypeError:
                n[i] = -1
        # All list arguments should have the same length
        mask = n != -1
        if len(n[mask]) and not np.all(n[mask] == n[mask][0]):
            raise Exception("All arguments given as lists must be of the same length")
        N = np.amax(n).astype(int)

        # Store the data, extending if necessary
        if N == -1:
            self.origins = np.column_stack((x, y, z)).astype(float)
            self.dirs = normalize_array(np.array([dir]).astype(float))
            self.weights = np.array([weight]).astype(float)
        else:
            if n[0] == -1:
                x = np.array([x]*N)
            if n[1] == -1:
                y = np.array([y]*N)
            if n[2] == -1:
                z = np.array([z]*N)
            if n[3] == -1:
                try:
                    len(dir)
                except TypeError:
                    raise Exception("dir must be either a vector or a list of vectors")
                dir = np.array([dir]*N)
            if n[4] == -1:
                weight = np.array([weight]*N)
            self.origins = np.column_stack((x, y, z)).astype(float)
            self.dirs = normalize_array(np.array(dir)).astype(float)
            self.weights = np.array(weight).astype(float)


class AdaptiveGaussianRF(RayFactory):
    def __init__(self, waist_origin, dir, waist_radius, power, n, wavelength, origin, emit_radius, curve=False, random_switch=False):
        super().__init__()
        # Calculate the ray origin distribution
        N = int(n)
        R = emit_radius

        # n is the number of rings, d is the distance between rings,
        # d_ring the distance between points on the rings (roughly)
        # We aim for d and d_ring to be similar
        n = round(.5 * (1 + np.sqrt(1 + 4 * (N - 1) / np.pi)))
        d = R / (n - .5)
        d_ring = np.pi * d * n * (n - 1) / (N - 1)

        # Distribute the points
        os = np.zeros((N, 3))
        areas = np.zeros(N)
        areas[0] = np.pi * (d/2)**2
        start = 1
        offset = 0
        for i in range(1, n):
            n_ring = round((2 * np.pi * i * d + offset) / d_ring)
            offset = (2 * np.pi * i * d + offset) - n_ring * d_ring
            t_ring = np.arange(0, 2 * np.pi, 2 * np.pi / n_ring)
            if random_switch:
                t_ring += 2*np.pi*random()
            os[start:start+n_ring] = np.array((i*d*np.cos(t_ring), i*d*np.sin(t_ring), np.zeros(n_ring))).T
            areas[start:start + n_ring] = 2*i*np.pi*d**2 / n_ring
            start += n_ring

        # Rotate the points according to the dir vector
        _dir = normalize(np.array(dir))
        os = rot_to_vector(os, _dir)

        # Move the calculated points to the emit origin
        self.origins = os + origin

        self.dirs = np.full((N, 3), _dir)

        # Calculate the necessary ray weights
        z = _dir.dot(np.array(origin) - np.array(waist_origin))
        rv = self.origins - np.array(waist_origin)
        rvr = rv - np.einsum("ij,j->i", rv, _dir).reshape((-1,1)) * _dir
        r = np.sqrt(np.einsum('...i,...i', rvr, rvr))

        photon_energy = 6.62607004e-25 * 299792458 / wavelength
        w = waist_radius * np.sqrt(1 + ((z*wavelength*1e-9)/(np.pi*waist_radius**2))**2)
        intensities = 2*power / (np.pi*w**2) * np.exp(-2 * r**2 / w ** 2)
        self.weights = intensities / photon_energy * areas

        # Calculate directions for spreading rays
        if curve:
            z_r = np.pi*waist_radius**2/(wavelength*1e-9)
            a = r * z / (z ** 2 + z_r ** 2)
            a[a>1] = 1
            b = np.sqrt(1 - a**2)
            rvr_n = normalize_array(rvr)
            self.dirs = np.einsum("i,j->ji", _dir, b) + np.einsum("ij,i->ij", rvr_n, a)
        else:
            self.dirs = np.full((N, 3), _dir)

        self.wavelength = wavelength


class HexagonalGaussianRF(RayFactory):
    def __init__(self, waist_origin, dir, waist_radius, power, n, wavelength, origin, emit_radius, curve=False):
        super().__init__()
        # Calculate the ray origin distribution
        n = int(np.round((3 + np.sqrt(12 * n - 3)) / 6))
        N = 3*n*(n-1) + 1
        r = emit_radius

        # a is the number of rings at the widest point
        a = 2 * n - 1
        b = 2 * r / np.sqrt(3) / (n - 1 / 3)

        # Distribute the points
        x = []
        y = []
        for j in range(-n + 1, n):
            for i in range(-(a // 2), a // 2 - abs(j) + 1):
                x.append(b * (i + abs(j) / 2))
                y.append(b * (np.sqrt(3) * j / 2))
        os = np.column_stack((x, y, np.zeros_like(x)))

        # Rotate the points according to the dir vector
        _dir = normalize(np.array(dir))
        os = rot_to_vector(os, _dir)

        # Move the calculated points to the emit origin
        self.origins = os + origin

        self.dirs = np.full((N, 3), _dir)

        # Calculate the necessary ray weights
        z = _dir.dot(np.array(origin) - np.array(waist_origin))
        rv = self.origins - np.array(waist_origin)
        rvr = rv - np.einsum("ij,j->i", rv, _dir).reshape((-1,1)) * _dir
        r = np.sqrt(np.einsum('...i,...i', rvr, rvr))

        photon_energy = 6.62607004e-25 * 299792458 / wavelength
        w = waist_radius * np.sqrt(1 + ((z*wavelength*1e-9)/(np.pi*waist_radius**2))**2)
        intensities = 2*power / (np.pi*w**2) * np.exp(-2 * r**2 / w ** 2)
        self.weights = intensities / photon_energy * np.sqrt(3)*b**2/2

        # Calculate directions for spreading rays
        if curve:
            z_r = np.pi*waist_radius**2/(wavelength*1e-9)
            a = r * z / (z ** 2 + z_r ** 2)
            a[a>1] = 1
            b = np.sqrt(1 - a**2)
            rvr_n = normalize_array(rvr)
            self.dirs = np.einsum("i,j->ji", _dir, b) + np.einsum("ij,i->ij", rvr_n, a)
        else:
            self.dirs = np.full((N, 3), _dir)

        self.wavelength = wavelength


##################################
#     Specific TracerObjects     #
##################################
class Surface(TracerObject):
    """
    An infinite surface.
    """
    def __init__(self, origin, normal, *args, **kwargs):
        """
        Create a surface with an origin and a normal.

        :param origin: np.array, [X,Y]
        :param normal: np.array, [X,Y] for the plane's normal vector. Doesn't have to be a unit vector
        :param kwargs: TracerObject's kwargs
        """
        super().__init__(origin, *args, **kwargs)
        self._normal = jm.rotate(np.array([normalize(np.array(normal))]), self._rot)[0]
        self.along = np.array([self._normal[1], -self._normal[0]]).astype(float)

    def normals(self, points):
        return np.full((len(points), 3), self._normal)

    def intersect_d(self, os, dirs):
        dot = np.einsum("ij,j->i", dirs, self._normal)
        d = np.full(len(os), np.inf)
        # We're being nice and avoiding division by 0
        d[dot != 0] = np.einsum("ij,j->i", (self.origin - os[dot != 0]), self._normal) / dot[dot != 0]
        # Only return positive distances
        d[d < 0] = np.inf
        return d

    def plot(self, ax):
        points = np.array([[-.5, .5, 0],
                           [.5, .5, 0],
                           [.5, -.5, 0],
                           [-.5, -.5, 0],
                           [-.5, .5, 0]])
        points = rot_to_vector(points, self._normal) + self.origin

        ax.plot(points[:, 0], points[:, 1], points[:, 2])
        ax.quiver(*self.origin, *(self._normal*0.1), color="tab:orange")


class Sphere(TracerObject):
    """
    A Sphere.
    """
    def __init__(self, origin, radius, *args, **kwargs):
        """
        Create a sphere or a section of it.

        :param origin: np.array, [X,Y]
        :param radius: float representing the sphere's radius
        :param kwargs: TracerObject's kwargs
        """
        super().__init__(origin, *args, **kwargs)
        self.radius = radius

    def normals(self, points):
        return normalize_array(points - self.origin)

    def intersect_d(self, os, dirs):
        # Compute d from some vector maths
        r = os - self.origin
        a = -np.einsum("ij,ij->i", r, dirs)
        b = a**2 - np.einsum('...i,...i', r, r) + self.radius**2
        b[b < 0] = np.inf
        b = np.sqrt(b)
        negative = a - b
        n_mask = negative > 0
        positive = a + b
        p_mask = positive > 0
        d = np.full(len(os), np.inf)
        d[p_mask] = positive[p_mask]
        d[n_mask] = negative[n_mask]
        return d

    def plot(self, ax):
        ax.plot([self.origin[0]], [self.origin[1]], [self.origin[2]], "o", ms=10, alpha=0.2)


class Triangle(TracerObject):
    def __init__(self, origin, a, b, c, *args, **kwargs):
        origin = np.array([0, 0, 0]) if origin is None else np.array(origin)
        super().__init__(origin, *args, **kwargs)
        a, b, c = jm.rotate(np.array([a, b, c]), self._rot)
        self.a = self.origin + a
        self.b = self.origin + b
        self.c = self.origin + c
        self.edge1 = self.b-self.a
        self.edge2 = self.c-self.a
        self._normal = normalize(np.cross(self.edge1, self.edge2))

    def normals(self, points):
        return np.full((len(points), 3), self._normal)

    def intersect_d(self, os, dirs):
        return jm.intersect_d_triangles(os, dirs, self.a, self.edge1, self.edge2)

    def plot(self, ax):
        points = np.array((self.a, self.b, self.c, self.a)).T
        ax.plot(*points, c="tab:blue", alpha=0.5)
        points = np.array((self.a, self.a+self._normal*.1)).T
        ax.plot(*points, c="tab:orange", alpha=0.5)


class MeshTO(TracerObject):
    def __init__(self, origin, filename, scale, *args, **kwargs):
        super().__init__(origin, *args, **kwargs)
        with open(filename, 'r') as f:
            lines = f.readlines()
        verts = np.array([np.array(l[2:-1].split(" ")).astype(float) for l in lines if l[:2] == 'v '])
        verts = jm.rotate(verts, self._rot)
        verts = verts*scale + self.origin
        faces = np.array([[int(v.split("/")[0]) - 1 for v in l[2:-1].split(" ")] for l in lines if l[:2] == 'f '])
        self.scale = scale
        self.a = verts[faces[:, 0]]
        self.edge1 = verts[faces[:, 1]] - self.a
        self.edge2 = verts[faces[:, 2]] - self.a
        self._normals = normalize_array(np.cross(self.edge1, self.edge2))
        self._d = None

    def normals(self, points):
        normals = jm.mesh_normals(self._d, self._normals)
        return normals

    def intersect_d(self, os, dirs):
        md, d_all = jm.intersect_d_mesh(os, dirs, self.a, self.edge1, self.edge2)
        self._d = d_all
        return md

    def plot(self, ax):
        a = self.a
        b = a + self.edge1
        c = a + self.edge2
        for i in range(len(a)):
            points = np.array((a[i], b[i], c[i], a[i])).T
            ax.plot(*points, c="tab:blue", alpha=0.1)
            points = np.array((a[i], a[i]+self._normals[i]*.1*self.scale)).T
            ax.plot(*points, c="tab:orange", alpha=0.1)


class SmoothMeshTO(TracerObject):
    def __init__(self, origin, filename, scale, *args, **kwargs):
        super().__init__(origin, *args, **kwargs)
        with open(filename, 'r') as f:
            lines = f.readlines()
        verts = np.array([np.array(l[2:-1].split(" ")).astype(float) for l in lines if l[:2] == 'v '])
        verts = jm.rotate(verts, self._rot)
        verts = verts * scale + self.origin
        faces = np.array([[int(v.split("/")[0]) - 1 for v in l[2:-1].split(" ")] for l in lines if l[:2] == 'f '])

        vns = np.array([np.array(l[3:-1].split(" ")).astype(float) for l in lines if l[:3] == 'vn '])
        vns = jm.rotate(vns, self._rot)
        vni = np.array([[int(v.split("/")[2]) - 1 for v in l[2:-1].split(" ")] for l in lines if l[:2] == 'f '])
        self.na = normalize_array(vns[vni[:, 0]])
        self.nb = normalize_array(vns[vni[:, 1]])
        self.nc = normalize_array(vns[vni[:, 2]])
        self.scale = scale

        self.a = verts[faces[:, 0]]
        self.edge1 = verts[faces[:, 1]] - self.a
        self.edge2 = verts[faces[:, 2]] - self.a
        self._normals = None

    def normals(self, points):
        return self._normals

    def intersect_d(self, os, dirs):
        md, normals = jm.intersect_d_mesh_smooth(os, dirs, self.a, self.edge1, self.edge2, self.na, self.nb, self.nc)
        self._normals = normals
        # print(md)
        return md

    def plot(self, ax):
        a = self.a
        b = a + self.edge1
        c = a + self.edge2
        for i in range(len(a)):
            points = np.array((a[i], b[i], c[i], a[i])).T
            ax.plot(*points, c="tab:blue", alpha=0.2)
            points = np.array((a[i], a[i]+self.scale*.1*self.na[i])).T
            ax.plot(*points, c="tab:orange", alpha=.1)
            points = np.array((b[i], b[i] + self.scale*.1 * self.nb[i])).T
            ax.plot(*points, c="tab:orange", alpha=.1)
            points = np.array((c[i], c[i] + self.scale*.1 * self.nc[i])).T
            ax.plot(*points, c="tab:orange", alpha=.1)


############################
#     ObjectContainers     #
############################
class Mesh(ObjectContainer):
    def __init__(self, origin, filename, scale, *args, **kwargs):
        with open(filename, 'r') as f:
            lines = f.readlines()
        verts = np.array([np.array(l[2:-1].split(" ")).astype(float) for l in lines if l[:2] == 'v '])*scale
        faces = [[int(v.split("/")[0]) - 1 for v in l[2:-1].split(" ")] for l in lines if l[:2] == 'f ']
        objects = []
        for f in faces:
            objects.append(Triangle(origin, *verts[f], *args, **kwargs))
        super().__init__(objects)

