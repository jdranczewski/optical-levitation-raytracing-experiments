"""
A second version of a 2D ray tracing library, this time implemented with numpy arrays containing ray information.
This allows for very significant speedups, as numpy can process many rays in batches, with functions written in c.

Created during an Undergraduate Research Experience Programme placement at Imperial College London 2020
by Jakub Dranczewski.

There should be a Jupyter Notebook in this directory called "tdt2-experiments.ipynb" - this contains
examples of most things that can be done with this library.

To contact me, try (in no particular order)
* jbd17@ic.ac.uk (unless I left)
* jakub.dranczewski@gmail.com
* jdranczewski.github.io (there should be an email in the CV)
* some other social media platform

This code should also be available at https://github.com/jdranczewski/optical-levitation-raytracing-experiments

NOTE: All momenta values need to be multiplied by h (Planck's constant) * 1e9 (wavelength is stored in nm)
"""


import numpy as np
from matplotlib.patches import Circle, Wedge


############################
#     Helper functions     #
############################
def normalize(x):
    """
    Given any vector, return a vector of unit length pointing in the same direction.

    :param x: a vector in np.array form
    :return: np.array - normalised vector
    """
    return x / np.sqrt(x.dot(x))


def normalize_array(x):
    """
    Given an array of vectors, return an array of vectors of unit length pointing in the same direction.

    :param x: an array of vectors in np.array form
    :return: np.array - an array of normalised vectors
    """
    return x / np.sqrt(np.einsum('...i,...i', x, x)).reshape(-1, 1)


def linspace(a, b, N):
    """
    A version of np.linspace which tries to get even, and (most importantly) symmetric spacings between the
    returned numbers. Breaks down when a and b are different orders of magnitude. Does well when they're the same,
    unless the interval goes through 0, in which case the number distribution will be symmetric, but not exactly
    uniform. The distribution may not also reach a and b exactly, but will exactly respect N.

    :param a: Start of the interval
    :param b: End of the interval
    :param N: Number of points to return
    :return: np.array
    """
    d = (b-a)/(N-1)/2
    m = (b+a)/2
    if N % 2:
        return np.concatenate((np.arange(m, a-d, -2*d)[:0:-1], np.arange(m, b+d, 2*d)))
    else:
        return np.concatenate((np.arange(m-d, a-d, -2*d)[::-1], np.arange(m+d, b+d, 2*d)))


#################################
#     Main abstract classes     #
#################################
class Scene:
    """
    A container for a ray tracing situation.

    :parameter self.history: a series of snapshots of ray origins at each step.
    :parameter self.objects: a list of all the TracerObjects in the scene

    """

    def __init__(self, rf, objects=[]):
        """
        :param rf: a RayFactory object
        :param objects: a list of TracerObject objects
        """
        self.r_origins = rf.origins
        self.history = [self.r_origins.copy()]
        self.r_dirs = rf.dirs
        self.r_weights = rf.weights
        self.r_wavelength = rf.wavelength
        self.active = np.ones(len(self.r_origins)).astype(bool)
        self.objects = objects

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

    def run(self, limit=100, margin=1e-10, announce_steps=False):
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

    def plot(self, ax, ray_kwargs={}, m_quiver=False, m_quiver_kwargs={}, sparse=1):
        """
        Given a matplotlib axis object, plot all simulation elements onto it.

        :param ax: a matplotlib axis
        :param ray_kwargs: keyword arguments to pass to ax.plot when drawing rays
        :param m_quiver: if True, the changes of momentum are plotted for all TracerObjects in the scene
        :param m_quiver_kwargs: keyword arguments to pass to ax.quiver when drawing the momentum changes, most useful
                                example being "scale", which changes the size of the arrows
        :param sparse: integer, if specified to be n, only every nth ray will be drawn
        :return: None
        """
        # Not every ray has existed throughout the entire run, hence the great list comprehension below, which
        # constructs paths for all the rays.
        for i, ray_hist in enumerate(
                [[self.history[j][i] for j in range(len(self.history)) if i < len(self.history[j])] for i in
                 range(0, len(self.history[-1]), sparse)]):
            rh = np.array(ray_hist)
            ax.plot(rh[:, 0], rh[:, 1], alpha=self.r_weights[i], **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)


class TracerObject:
    """
    Base class for all objects that interact with rays in this ray tracer.

    :parameter self.origin: the object's origin
    :parameter self.momentum: total momentum accumulated by the object
    """
    def __init__(self, origin, n_out=1., n_in=1., reflective=False):
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
        self.origin = origin
        self.n_out = float(n_out)
        self.n_in = float(n_in)
        self.momentum = np.zeros(2)
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
        raise NotImplementedError

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
        cos_i = -np.einsum("ij,ij->i", dirs, normals)
        going_out = cos_i < 0
        n1 = np.full(len(os), self.n_out)
        n1[going_out] = self.n_in

        # Reflect the rays and store the momentum change
        change = 2 * np.einsum("ij,i->ij", normals, cos_i)
        self.momentum -= np.einsum("ij,i->j", change, n1*weights) / wavelength
        dirs += change

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

        # Establish normals, n1, and n2 arrays based on whether the rays are going in or out
        cos_i = -np.einsum("ij,ij->i", dirs, normals)
        going_out = cos_i < 0
        normals[going_out] *= -1
        n1 = np.full(len(os), self.n_out)
        n2 = np.full(len(os), self.n_in)
        n1[going_out] = self.n_in
        n2[going_out] = self.n_out
        # print(going_out, self.n_in, n1, n2)
        cos_i[going_out] *= -1

        # Calculate a few more angles
        sin_i2 = 1 - cos_i**2
        cos_t = np.sqrt(1 - (n1 / n2) ** 2 * (1 - cos_i ** 2))

        # Calculate the initial momentum of the rays
        init_weights = n1*weights

        # Mark out rays that undergo total internal reflection
        tir = np.sqrt(sin_i2) > n2 / n1
        ntir = np.invert(tir)
        # print("TIR", tir, ntir)

        # Calculate the reflection and refraction coefficients
        R_perp = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
        R_para = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
        R = (R_perp + R_para) / 2
        T = 1 - R

        # Calculate the ray's new direction under refraction and reflection
        d_refr = (np.einsum("i,ij->ij", n1/n2, dirs) +
                  np.einsum("i,ij->ij", (n1 / n2 * cos_i - np.sqrt(1 - (n1 / n2) ** 2 * sin_i2)), normals))
        d_refl = dirs + 2 * np.einsum("ij,i->ij", normals, cos_i)

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
        self.momentum -= (np.einsum("ij,i->j", d_refr, n2*weights) +
                          np.einsum("ij,i->j", new_d, n1[ntir]*new_weights) -
                          np.einsum("ij,i->j", dirs, init_weights)) / wavelength

        # For now return just the refraction
        return d_refr, weights, new_d, new_weights, new_origins

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis.

        :param ax: a matplotlib axis object
        :return: None
        """
        pass


#################################
#     Various Ray Factories     #
#################################
class RayFactoryLegacy:
    def __init__(self, rays):
        """
        Take in a set of Ray objects from two_d_tracer and make a RayFactory object.

        :param rays: a list of two_d_tracer.Ray objects
        """
        self.origins = np.array([ray.origin for ray in rays])
        self.dirs = np.array([ray.dir for ray in rays])
        self.weights = np.array([ray.weight for ray in rays]).astype(float)
        self.wavelength = rays[0].wavelength


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
        self._normal = normalize(np.array(normal))
        self.along = np.array([self._normal[1], -self._normal[0]])

    def normals(self, points):
        return np.full((len(points), 2), self._normal)

    def intersect_d(self, os, dirs):
        dot = np.einsum("ij,j->i", dirs, self._normal)
        d = np.full(len(os), np.inf)
        # We're being nice and avoiding division by 0
        d[dot != 0] = np.einsum("ij,j->i", (self.origin - os[dot != 0]), self._normal) / dot[dot != 0]
        # Only return positive distances
        d[d < 0] = np.inf
        return d

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis. In case of the Surface, the
        small triangle represents the direction of the normal.

        :param ax: a matplotlib axis object
        :return: None
        """
        points = np.array([self.origin + self.along, self.origin + 0.1 * self.along, self.origin + 0.3*self._normal,
                           self.origin - 0.1 * self.along, self.origin - self.along])
        ax.plot(points[:, 0], points[:, 1], ":")


class Sphere(TracerObject):
    """
    A Sphere.
    """
    def __init__(self, origin, radius, mask=None, *args, **kwargs):
        """
        Create a sphere or a section of it.

        :param origin: np.array, [X,Y]
        :param radius: float representing the sphere's radius
        :param mask: a subset of the circle that interacts with rays expressed as angles in radians. Zero is along x
                     axis, range is (-pi, pi]. If the angles provided (as a two-element list) make for a correct range,
                     that range is chosen. If they are in the other direction, the opposite range is chosen.
        :param kwargs: TracerObject's kwargs
        """
        super().__init__(origin, *args, **kwargs)
        self.radius = radius
        self.mask = mask if mask is None else np.array(mask)

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
        if self.mask is None:
            # Out of the two solutions, we care more about the smaller one (-), so let it override the + solution.
            d[p_mask] = positive[p_mask]
            d[n_mask] = negative[n_mask]
        else:
            # Mask the negative solutions
            d[n_mask] = negative[n_mask]
            r = os + np.einsum('ij,i->ij', dirs, d) - self.origin
            angle = np.arctan2(r[:, 1], r[:, 0])
            if self.mask[1] > self.mask[0]:
                out = np.logical_or(angle < self.mask[0], angle > self.mask[1])
            else:
                out = np.logical_and(angle < self.mask[0], angle > self.mask[1])
            d[out] = np.inf
            # Now go for the positive solutions
            # Conditions: no negative solution was found, and the positive solution is within the mask
            d[p_mask & out] = positive[p_mask & out]
            # The below is slightly inefficient, as we're testing some of the points we've already tested before
            r = os + np.einsum('ij,i->ij', dirs, d) - self.origin
            angle = np.arctan2(r[:, 1], r[:, 0])
            if self.mask[1] > self.mask[0]:
                out = np.logical_or(angle < self.mask[0], angle > self.mask[1])
            else:
                out = np.logical_and(angle < self.mask[0], angle > self.mask[1])
            d[out] = np.inf
        return d

    def plot(self, ax):
        if self.mask is None:
            patch = Circle(self.origin, self.radius, alpha=0.2)
        else:
            patch = Wedge(self.origin, self.radius, *self.mask*180/np.pi, 0.1, alpha=0.2)
        ax.add_artist(patch)


class LineSegment(Surface):
    def __init__(self, A, B, *args, **kwargs):
        """
        Create a refractive line segment

        :param A: Start point of the segment
        :param B: End point of the segment
        :param kwargs: Surface keyword arguments
        """
        self.A = np.array(A)
        self.B = np.array(B)
        along = normalize(self.A - self.B)
        normal = np.array([along[1], -along[0]])
        super().__init__(np.mean([A, B], axis=0), normal, *args, **kwargs)

    def intersect_d(self, os, dirs):
        # Compute ray normals
        rns = dirs[:, ::-1] * [1,-1]
        mask = np.einsum("ij,ij->i", rns, self.A-os) * np.einsum("ij,ij->i", rns, self.B-os) < 0
        d = np.full(len(os), np.inf)
        d[mask] = super().intersect_d(os[mask], dirs[mask])
        return d

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis.

        :param ax: a matplotlib axis object
        :return: None
        """
        stack = np.stack((self.A-0.2*self._normal, self.A, self.B, self.B-0.2*self._normal))
        ax.plot(stack[:, 0], stack[:, 1], ":")