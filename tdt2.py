import numpy as np
from matplotlib.patches import Circle, Wedge


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


class Scene:
    """
    A container for a ray tracing sitaution.
    """

    def __init__(self, rf, objects=[], wavelength=600):
        """
        :param rf: a RayFactory object
        :param objects: a list of TracerObject objects
        """
        self.r_origins = rf.origins
        self.history = [self.r_origins.copy()]
        self.r_dirs = rf.dirs
        self.r_weights = rf.weights
        self.r_wavelength = wavelength
        self.active = np.ones(len(self.r_origins)).astype(bool)
        self.objects = objects

    def step(self):
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
        new_dirs = np.array([])
        new_weights = np.array([])
        new_origins = np.array([])
        # Let objects do things to rays
        for i, obj in enumerate(self.objects):
            collided = collisions == i
            self.r_dirs[collided], self.r_weights[collided], nd, nw, no = \
                obj.act_rays(self.r_origins[collided], self.r_dirs[collided],
                             self.r_weights[collided], self.r_wavelength)
            # print("NEW", nd)
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
                 range(len(self.history[-1]))]):
            rh = np.array(ray_hist)
            ax.plot(rh[:, 0], rh[:, 1], alpha=self.r_weights[i], **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)


class RayFactoryLegacy:
    def __init__(self, rays):
        """
        Take in a set of Ray objects from two_d_tracer and make a RayFactory
        :param rays:
        """
        self.origins = np.array([ray.origin for ray in rays])
        self.dirs = np.array([ray.dir for ray in rays])
        self.weights = np.array([ray.weight for ray in rays]).astype(float)


class TracerObject:
    def __init__(self, origin, n_out=1., n_in=1.):
        self.origin = origin
        self.n_out = float(n_out)
        self.n_in = float(n_in)

    def intersect_d(self, os, dirs):
        raise NotImplementedError

    def normals(self, points):
        raise NotImplementedError

    def reflect(self, ray, point):
        raise NotImplementedError

    def refract(self, ray, point):
        raise NotImplementedError

    def act_rays(self, os, dirs, weights, wavelength):
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

        # For now return just the refraction
        return d_refr, weights, new_d, new_weights, new_origins

    def plot(self, ax):
        pass

    @property
    def momentum(self):
        raise NotImplementedError

    @property
    def angular_momentum(self, origin=None):
        raise NotImplementedError


class Surface(TracerObject):
    def __init__(self, origin, normal, *args, **kwargs):
        super().__init__(origin, *args, **kwargs)
        self._normal = normalize(np.array(normal))
        self.along = np.array([self._normal[1], -self._normal[0]])

    def normals(self, points):
        return np.full((len(points), 2), self._normal)

    def intersect_d(self, os, dirs):
        dot = np.einsum("ij,j->i", dirs, self._normal)
        d = np.full(len(os), np.inf)
        d[dot != 0] = np.einsum("ij,j->i", (self.origin - os[dot != 0]), self._normal) / dot[dot != 0]
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
    def __init__(self, origin, radius, *args, **kwargs):
        super().__init__(origin, *args, **kwargs)
        self.origin = origin
        self.radius = radius

    def normals(self, points):
        return normalize_array(points - self.origin)

    def intersect_d(self, os, dirs):
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
        patch = Circle(self.origin, self.radius, alpha=0.2)
        ax.add_artist(patch)
