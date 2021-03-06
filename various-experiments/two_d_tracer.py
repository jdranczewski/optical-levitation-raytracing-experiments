"""
This is an early version of a raytracing library, for now working only in 2 dimensions, created during
an Undergraduate Research Experience Programme placement at Imperial College London 2020 by Jakub Dranczewski.

There should be a Jupyter Notebook in this directory called "two-d-tracer-experiments.ipynb" - this contains
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
from matplotlib.colors import hsv_to_rgb
from scipy.special import erfinv
import random


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


def nm_to_rgb(wvl, margin=30):
    """
    A simple conversion from wavelength (in nanometers) to a corresponding RGB value.
    Not completely physically correct, just for display purposes.

    :param wvl: Wavelength in nanometers
    :param margin: easing at the ends of the spectrum towards invisible light (in nm)
    :return: np.array - (R, G, B), where values are from 0 to 1
    """
    wv = np.array([380, 460, 480, 515, 590, 630, 670])
    hv = np.array([197, 174, 135, 89, 42, 23, 0])/255
    h = np.interp(wvl, wv, hv)
    v = np.interp(wvl, [380, 380+margin, 740-margin, 740], [0, 1, 1, 0])
    return hsv_to_rgb((h, 1, v))


def seed(s):
    """
    Seed two_d_tracer's random library's instance.

    :param s: seed
    :return: None
    """
    random.seed(s)


def gaussian_randoms_factory(w):
    """
    A factory for making random number generators which follow the gaussian beam intensity distribution
    (in terms of radius)

    :param w: width of the beam
    :return: a random number generating function
    """
    return lambda: w/np.sqrt(2)*erfinv(random.random())


def random_sign():
    return 2*(random.randint(0, 1) - 0.5)


def gaussian_intensity_factory(total_power, w, origin):
    """
    This returns a function that gives the intensity of a Gaussian beam with given properties at a certain distance.
    Note that this is normalised for the 2D case (so integrating in one direction gives total_power.

    :param total_power: Total power in the beam
    :param w: radius of the beam (defined as usual for Gaussian beams)
    :param origin: [X, Y] - origin of the distribution in 2D space - note that this is separate from the RayBundle
                   origin to allow for sampling parts of the beam
    :return: float
    """
    return lambda pos: np.sqrt(2/np.pi)* total_power/w * np.exp(-2*(pos-origin).dot(pos-origin)/w**2)


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
    """
    def __init__(self, rays, objects):
        """
        :param rays: a list of Ray objects
        :param objects: a list of TracerObject objects
        """
        self.rays = rays
        self.objects = objects

    def step(self):
        """
        Make a single ray tracing step - each ray moves towards its next collision.

        :return: None
        """
        new_rays = []
        for ray in self.rays:
            if not ray.done:
                # Find the next intersection for each ray
                inter, obj = self.intersect(ray)
                # If no intersection found, terminate ray
                if obj:
                    nr = obj.act_ray(ray, inter)
                    if nr:
                        new_rays.append(nr)
                else:
                    ray.stop()
        self.rays += new_rays

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
            self.propagate(margin)
            if announce_steps:
                print("Step", i)
            if all([ray.done for ray in self.rays]):
                break

    def plot(self, ax, true_color=True, ray_kwargs={}, m_quiver=False, m_quiver_kwargs={}, sparse=1):
        """
        Given a matplotlib axis object, plot all simulation elements onto it.

        :param ax: a matplotlib axis
        :param true_color: if True, the ray's actual colour used for plotting. Otherwise matplotlib handles colours
        :param ray_kwargs: keyword arguments to pass to ax.plot when drawing rays
        :param m_quiver: if True, the changes of momentum are plotted for all TracerObjects in the scene
        :param m_quiver_kwargs: keyword arguments to pass to ax.quiver when drawing the momentum changes, most useful
                                example being "scale", which changes the size of the arrows
        :param sparse: integer, if specified to be n, only every nth ray and quiver arrow is drawn
        :return: None
        """
        for ray in self.rays[::sparse]:
            ax.plot(ray.history[:, 0], ray.history[:, 1], c=ray.c if true_color else None,
                    alpha=ray.alpha, **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)
        if m_quiver:
            ax.quiver(self.m_pos[::sparse, 0], self.m_pos[::sparse, 1], self.momenta[::sparse, 0],
                      self.momenta[::sparse, 1], **m_quiver_kwargs)

    def intersect(self, ray):
        """
        Find the nearest object a ray will encounter

        :param ray: a Ray object
        :return: distance to the nearest object, the nearest object
        """
        d_min = np.inf
        obj_min = None
        for obj in self.objects:
            d = obj.intersect_d(ray)
            if d < d_min:
                d_min = d
                obj_min = obj
        if obj_min:
            return ray.origin + d_min*ray.dir, obj_min
        else:
            return np.inf, obj_min

    def propagate(self, d):
        """
        Propagate the rays along their current direction. Doesn't take collisions into account.

        :param d: distance to propagate
        :return: None
        """
        for ray in self.rays:
            ray.propagate(d)

    @property
    def momenta(self):
        """
        np.array of [X, Y] vectors representing the changes of momentum for all TracerObjects in the scene

        :return: np.array
        """
        return np.concatenate([obj.momenta for obj in self.objects if len(obj.momenta)])

    @property
    def m_pos(self):
        """
        np.array of [X, Y] vectors representing the positions of changes of momentum for all TracerObjects in the scene

        :return: np.array
        """
        return np.concatenate([obj.m_pos for obj in self.objects if len(obj.m_pos)])

    @property
    def momentum(self):
        """
        Total momentum acquired by all TracerObjects within the scene.

        :return: float
        """
        return np.sum(self.momenta, axis=0)


class Ray:
    """
    A ray for the ray tracing simulation.
    """
    def __init__(self, origin, direction, wavelength=467, weight=1, max_weight=None, c=None):
        """
        Create a new ray.

        :param origin: 1D, 2-element list describing the ray's origin in 2D space (x,y)
        :param direction: A unit vector pointed in the ray's propagation direction
        :param wavelength: the ray's wavelength in nanometers
        :param weight: the number of real photons this ray corresponds to. Used in momentum calculations
        """
        self._origin = np.array(origin)
        self._history = [self._origin]
        self.dir = normalize(np.array(direction))
        self.wavelength = wavelength
        self.c = c if c is not None else nm_to_rgb(wavelength)
        self.weight = weight
        self.max_weight = max_weight if max_weight else weight
        self.done = False

    def stop(self):
        """
        Terminate the ray

        :return: None
        """
        self.done = True
        self.propagate(1)

    def propagate(self, d):
        """
        Move the origin along the ray's direction. This doesn't take collisions into account.

        :param d: distance to move the origin by
        :return: None
        """
        self.origin += self.dir * d

    @property
    def origin(self):
        """
        The current origin of the ray.

        :return: np.array(X, Y)
        """
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        self._origin = np.array(new_origin)
        self._history.append(self._origin)

    @property
    def history(self):
        """
        The points at which this ray has been.

        :return: np.array: [[X,Y], [X,Y], ...]
        """
        return np.array(self._history)

    @property
    def normal(self):
        """
        A vector normal to the ray's direction.

        :return: np.array - [X, Y]
        """
        return self.dir[::-1] * [1, -1]

    @property
    def angle(self):
        """
        The direction of the ray described as an angle in radians, where 0 corresponds to
        the ray moving along the x axis.

        :return: a float representing the angle in radians
        """
        return np.arctan2(*self.dir[::-1])

    @property
    def alpha(self):
        """
        The alpha (transparency) at whihc this ray should be rendered.

        :return: float [0,1]
        """
        return self.weight/self.max_weight

    def __repr__(self):
        return "Ray({}, {})".format(self._origin, self.dir)


class RayBundle:
    def __init__(self, origin, dir, radius, n, wavelength, intensity, label=None):
        self.origin = np.array(origin)
        self.dir = np.array(dir)
        self.normal = self.dir[::-1] * [1,-1]
        self.wavelength = wavelength
        self.label = label
        c = nm_to_rgb(wavelength)
        # Generate the rays
        self.rays = []
        spacing = 2*radius/n
        photon_energy = 6.62607004e-25 * 299792458/wavelength
        for i in range(n):
            pos = self.origin + (i-(n-1)/2)*spacing*self.normal
            weight = intensity(pos) * spacing / photon_energy
            self.rays.append(Ray(pos, dir, wavelength, weight,c=c))
        max_weight = np.amax([ray.weight for ray in self.rays])
        for ray in self.rays:
            ray.max_weight = max_weight

    def __repr__(self):
        return ("{}: ".format(self.label) if self.label else "") + "RayBundle({} rays)".format(len(self.rays))


class RandomRayBundle:
    def __init__(self, origin, dir, n, energy, wavelength, r_generator, theta_generator, label=None):
        self.origin = np.array(origin)
        self.dir = np.array(dir)
        self.normal = self.dir[::-1] * [1,-1]
        self.energy = energy
        self.wavelength = wavelength
        self.label = label
        # Calculate ray weights
        photon_number = energy / (6.62607004e-25 * 299792458/wavelength)
        weight = photon_number / n
        # Generate the rays
        pos = lambda: self.origin + self.normal * r_generator() * theta_generator()
        self.rays = [Ray(pos(), dir, wavelength, weight) for i in range(n)]

    def __repr__(self):
        return ("{}: ".format(self.label) if self.label else "") + "RandomRayBundle({} rays)".format(len(self.rays))


class TracerObject:
    """
    Base class for all objects that interact with rays in this ray tracer.
    """
    def __init__(self, origin, n_out=1, n_in=1):
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
        """
        self.origin = origin
        self.n_out = n_out
        self.n_in = n_in
        self.momenta = []
        self.m_pos = []

    def intersect_d(self, ray):
        """
        Given a Ray object, return the distance the ray has to travel before intersecting this object.

        :param ray: a Ray object
        :return: float, distance the ray has to travel to intersect, must be positive
        """
        raise NotImplementedError

    def normal(self, point):
        """
        Calculate the normal vector to this object's surface at a given point

        :param point: [X, Y] coordinates of the point
        :return: np.array, the normal vector, normalised
        """
        raise NotImplementedError

    def reflect(self, ray, point):
        """
        Modify the given ray as a result of reflection at a given point.

        :param ray: a Ray object
        :param point: [X, Y] coordinates of the point
        :return: None
        """
        # Check which refractive index we should be working with
        if np.dot(ray.dir, self.normal(point)) < 0:
            n = self.n_out(ray.wavelength) if callable(self.n_out) else self.n_out
        else:
            n = self.n_in(ray.wavelength) if callable(self.n_in) else self.n_in

        ray.origin = point
        change = 2 * self.normal(point) * np.dot(self.normal(point), ray.dir)
        ray.dir -= change

        # Calculate the change in momentum
        self.momenta.append(change * n / ray.wavelength * ray.weight)
        self.m_pos.append(point)

    def refract(self, ray, point):
        """
        Modify the given ray as a result of refraction at a given point.

        :param ray: a Ray object
        :param point: [X, Y] coordinates of the point
        :return: None
        """
        ray.origin = point

        # Check what direction the light's going
        n1 = self.n_out(ray.wavelength) if callable(self.n_out) else self.n_out
        n2 = self.n_in(ray.wavelength) if callable(self.n_in) else self.n_in
        normal = self.normal(point)
        if np.dot(ray.dir, self.normal(point)) > 0:
            n1, n2 = n2, n1
            normal = -normal

        # Check for total internal reflection
        cos_i = -np.dot(ray.dir, normal)
        sin_i2 = 1 - cos_i**2
        if np.sqrt(sin_i2) > n2/n1:
            self.reflect(ray, point)
        else:
            # Calculate the reflection and refraction coefficients
            cos_t = np.sqrt(1 - (n1/n2)**2*(1-cos_i**2))
            R_perp = ((n1*cos_i-n2*cos_t) / (n1*cos_i+n2*cos_t))**2
            R_para = ((n2*cos_i-n1*cos_t) / (n2*cos_i+n1*cos_t))**2
            R = (R_perp + R_para) / 2
            T = 1 - R
            # Avoid creating rays at a very low weight
            if R*ray.weight/ray.max_weight < 1e-5:
                T, R = 1, 0
            else:
                # Make a copy of the original ray
                ray_reflected = Ray(point, ray.dir, ray.wavelength, ray.weight, ray.max_weight, ray.c)
            # Calculate the original momentum of the refracted ray portion
            m_init = ray.dir * n1 / ray.wavelength
            # Refract the original ray
            ray.dir = n1/n2*ray.dir + (n1/n2*cos_i - np.sqrt(1 - (n1/n2)**2 * sin_i2)) * normal
            ray.weight *= T
            self.momenta.append((m_init - ray.dir * n2 / ray.wavelength) * ray.weight)
            self.m_pos.append(point)
            if R:
                # Reflect the new ray
                ray_reflected.weight *= R
                self.reflect(ray_reflected, point)
                return ray_reflected

    def act_ray(self, ray, point):
        """
        Given a ray object, interact with it (could be a reflection or refraction, or some other action).

        :param ray: a Ray object
        :param point: [X, Y] coordinates of the point of interaction
        :return: None, or any new rays created as a result of this interaction
        """
        raise NotImplementedError

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis.

        :param ax: a matplotlib axis object
        :return: None
        """
        pass

    @property
    def momentum(self):
        """
        Total momentum acquired by this object

        :return: np.array - [X,Y]
        """
        return np.sum(self.momenta, axis=0)

    @property
    def angular_momentum(self, origin=None):
        """
        Total angular momentum acquired by this object around its origin or a given origin.

        :param origin: If given, the angular momentum will be calculated around this point.
        :return: float, signed according to the right hand rule.
        """
        o = self.origin if origin is None else origin
        r = np.array(self.m_pos) - o
        p = np.array(self.momenta)
        am = r[:, 0] * p[:, 1] - r[:, 1] * p[:, 0]
        return np.sum(am)


class ObjectContainer:
    def __init__(self, objects, label=None):
        """
        A container for objects. Can be used to create more complicated systems (like a prism with three LineSegments),
        allowing for better labelling and easier momentum inspection.

        :param objects: A list of objects that go in the container
        :param label: A label which will be used in the objects str() representation
        """
        self.objects = objects
        self.label = label

    @property
    def momenta(self):
        """
        np.array of [X, Y] vectors representing the changes of momentum for all TracerObjects in the Container

        :return: np.array
        """
        return np.concatenate([obj.momenta for obj in self.objects if len(obj.momenta)])

    @property
    def m_pos(self):
        """
        np.array of [X, Y] vectors representing the positions of changes of momentum
        for all TracerObjects in the Container

        :return: np.array
        """
        return np.concatenate([obj.m_pos for obj in self.objects if len(obj.m_pos)])

    @property
    def momentum(self):
        """
        np.array of [X, Y] vectors representing the total change of momentum for all TracerObjects in the Container

        :return: np.array
        """
        return np.sum(self.momenta, axis=0)

    def __repr__(self):
        return ("{}: ".format(self.label) if self.label else "") + "ObjectContainer({})".format(self.objects)


##################################
#     Specific TracerObjects     #
##################################
class Surface(TracerObject):
    def __init__(self, origin, normal, radius=None, *args, **kwargs):
        """
        Create a new Plane Object.

        :param origin: Coordinates of the object's centre - [X, Y]
        :param normal: A vector normal to the plane, [X, Y], doesn't have to be normalised
        :param radius: distance from the origin over which the plane interacts with rays
        :param n_out: Refractive index outside of the object
        :param n_in: Refractive index inside of the object
        """
        super().__init__(origin, *args, **kwargs)
        self._normal = normalize(np.array(normal))
        self.along = np.array([self._normal[1], -self._normal[0]])
        self.radius = radius

    def intersect_d(self, ray):
        dot = np.dot(ray.dir, self._normal)
        if dot != 0:
            d = np.dot(self.origin - ray.origin, self._normal) / dot
            return d if d > 0 else np.inf
        else:
            return np.inf

    def normal(self, point):
        return self._normal

    def act_ray(self, ray, point):
        if self.radius is None or np.linalg.norm(point - self.origin) <= self.radius:
            return self.refract(ray, point)
        else:
            ray.origin = point

    def __repr__(self):
        return "Surface({}, {})".format(self.origin, self._normal)

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis. In case of the Surface, the
        small triangle represents the direction of the normal.

        :param ax: a matplotlib axis object
        :return: None
        """
        if self.radius is None:
            points = np.array([self.origin + self.along, self.origin + 0.1 * self.along, self.origin + 0.3*self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.along])
        else:
            points = np.array([self.origin + self.radius*self.along, self.origin + 0.1 * self.along, self.origin + 0.3*self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.radius*self.along])
        ax.plot(points[:, 0], points[:, 1], ":")


class SurfaceReflective(Surface):
    def act_ray(self, ray, point):
        if self.radius is None or np.linalg.norm(point - self.origin) <= self.radius:
            return self.reflect(ray, point)
        else:
            ray.origin = point

    def __repr__(self):
        return "Mirror({}, {})".format(self.origin, self._normal)


class LineSegment(Surface):
    def __init__(self, A, B, *args, **kwargs):
        """
        Create a refractive line segment

        :param A: Start point of the segment
        :param B: End point of the segment
        :param args: Surface arguments
        :param kwargs: Surface keyword arguments (note: radius is not supported by this object)
        """
        self.A = np.array(A)
        self.B = np.array(B)
        along = normalize(self.A - self.B)
        normal = np.array([along[1], -along[0]])
        super().__init__(np.mean([A, B], axis=0), normal, None, *args, **kwargs)

    def intersect_d(self, ray):
        r_n = ray.normal
        if np.dot(r_n, self.A-ray.origin) * np.dot(r_n, self.B-ray.origin) < 0:
            return super().intersect_d(ray)
        else:
            return np.inf

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis.

        :param ax: a matplotlib axis object
        :return: None
        """
        stack = np.stack((self.A-0.2*self._normal, self.A, self.B, self.B-0.2*self._normal))
        ax.plot(stack[:, 0], stack[:, 1], ":")


class LineSegmentReflective(LineSegment):
    def act_ray(self, ray, point):
        return self.reflect(ray, point)


class RayCanvas(Surface):
    """
    A transparent plane object that records data about the rays passing through it.
    Access .points, .wavelengths, and .c (colour) lists to get the data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = []
        self.wavelengths = []
        self.c = []
        self.weights = []

    def act_ray(self, ray, point):
        ray.origin = point
        self.points.append(np.dot(self.along, point))
        self.wavelengths.append(ray.wavelength)
        self.c.append(ray.c)
        self.weights.append(ray.weight)

    def __repr__(self):
        return "RayCanvas({}, {}): {}".format(self.origin, self._normal, self.points)

    def plot(self, ax):
        if self.radius is None:
            points = np.array([self.origin + self.along, self.origin + 0.1 * self.along, self.origin + 0.3*self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.along])
        else:
            points = np.array([self.origin + self.radius*self.along, self.origin + 0.1 * self.along, self.origin + 0.3*self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.radius*self.along])
        ax.plot(points[:, 0], points[:, 1], "--")

    @property
    def alphas(self):
        return self.weights/np.amax(self.weights)

    @property
    def ca(self):
        """
        :return: np.array, with [R,G,B,alpha] data for all points
        """
        return np.column_stack((np.array(self.c), self.alphas))


class Sphere(TracerObject):
    """
    A TracerObject shaped like a sphere
    """
    def __init__(self, origin, radius, mask=None, *args, **kwargs):
        """
        Create a new Sphere Object.

        :param origin: coordinates of the object's centre - [X, Y]
        :param radius: radius of the sphere
        :param mask: a subset of the circle that interacts with rays expressed as angles in radians. Zero is along x
                     axis, range is (-pi, pi]. If the angles provided (as a two-element list) make for a correct range,
                     that range is chosen. If they are in the other direction, the opposite range is chosen.
        :param n_out: Refractive index outside of the object
        :param n_in: Refractive index inside of the object (both optional, default to 1)
        """
        super().__init__(origin, *args, **kwargs)
        self.radius = radius
        self.mask = None if mask is None else np.array(mask)

    def intersect_d(self, ray):
        r = ray.origin - self.origin
        a = -np.dot(r, ray.dir)
        b = a**2 - r.dot(r) + self.radius**2
        if b < 0:
            return np.inf
        b = np.sqrt(b)
        if a > b:
            return a-b
        elif a + b > 0:
            return a+b
        else:
            return np.inf

    def normal(self, point):
        return normalize(point-self.origin)

    def act_ray(self, ray, point):
        # print(np.arctan2(*(point - self.origin)[::-1])*180/np.pi)
        angle = np.arctan2(*(point - self.origin)[::-1])
        if (self.mask is None or
                (self.mask[0] < self.mask[1] and self.mask[0] <= angle <= self.mask[1]) or
                (self.mask[0] > self.mask[1] and (angle >= self.mask[0] or angle <= self.mask[1]))):
            return self.refract(ray, point)
        else:
            ray.origin = point

    def plot(self, ax):
        if self.mask is None:
            patch = Circle(self.origin, self.radius, alpha=0.2)
        else:
            patch = Wedge(self.origin, self.radius, *self.mask*180/np.pi, 0.1, alpha=0.2)
        ax.add_artist(patch)

    def __repr__(self):
        return "Sphere({}, {})".format(self.origin, self.radius)


class SphereReflective(Sphere):
    def act_ray(self, ray, point):
        angle = np.arctan2(*(point - self.origin)[::-1])
        if (self.mask is None or
                (self.mask[0] < self.mask[1] and self.mask[0] <= angle <= self.mask[1]) or
                (self.mask[0] > self.mask[1] and (angle >= self.mask[0] or angle <= self.mask[1]))):
            return self.reflect(ray, point)
        else:
            ray.origin = point


class Parabola(TracerObject):
    def __init__(self, a, b, c, xrange=(-5, 5), *args, **kwargs):
        super().__init__([0, 0], *args, **kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.xrange = xrange

    def intersect_d(self, ray):
        P_x, P_y = ray.origin
        D_x, D_y = ray.dir
        a, b, c = self.a, self.b, self.c
        if not D_x:
            d = (P_x**2*a + P_x*b - P_y + c)/D_y
            if d > 0:
                return d
            else:
                return np.inf
        d = -(2 * D_x * P_x * a + D_x * b - D_y) / (2 * D_x ** 2 * a) - np.sqrt(
            4 * D_x ** 2 * P_y * a - 4 * D_x ** 2 * a * c + D_x ** 2 * b ** 2 - 4 * D_x * D_y * P_x * a - 2 * D_x * D_y * b + D_y ** 2) / (
                    2 * D_x ** 2 * a)
        if d > 0:
            return d
        d = -(2*D_x*P_x*a + D_x*b - D_y)/(2*D_x**2*a) + np.sqrt(4*D_x**2*P_y*a - 4*D_x**2*a*c + D_x**2*b**2 - 4*D_x*D_y*P_x*a - 2*D_x*D_y*b + D_y**2)/(2*D_x**2*a)
        if d > 0:
            return d
        return np.inf

    def normal(self, point):
        alpha = np.arctan(2*self.a*point[0]+self.b)
        return np.array([np.sin(alpha), -np.cos(alpha)])

    def act_ray(self, ray, point):
        if self.xrange[0] <= point[0] <= self.xrange[1]:
            return self.refract(ray, point)
        else:
            ray.origin = point

    def plot(self, ax):
        x = np.linspace(*self.xrange, 100)
        ax.plot(x, self.a*x**2 + self.b*x + self.c, ":")

    def __repr__(self):
        return "Parabola({}, {}, {)".format(self.a, self.b, self.c)


class ParabolaReflective(Parabola):
    def act_ray(self, ray, point):
        if self.xrange[0] <= point[0] <= self.xrange[1]:
            return self.reflect(ray, point)
        else:
            ray.origin = point