"""
This is an early version of a raytracing library, for now working only in 2 dimensions.
"""
import numpy as np
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import hsv_to_rgb


def normalize(x):
    """
    Given any vector, return a vector of unit length pointing in the same direction.

    :param x: a vector in array/list form
    :return: np.array - normalised vector
    """
    x = x/np.linalg.norm(x)
    return x


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
    v = np.interp(wvl, [380, 380+margin, 740-margin, 740], [0,1,1,0])
    return hsv_to_rgb((h, 1, v))


class Scene:
    """
    A container for a ray tracing sitaution.
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
        for ray in self.rays:
            if not ray.done:
                # Find the next intersection for each ray
                inter, obj = self.intersect(ray)
                # If no intersection found, terminate ray
                if obj:
                    obj.act_ray(ray, inter)
                else:
                    ray.stop()

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

    def plot(self, ax, true_color=True, ray_kwargs={}):
        """
        Given a matplotlib axis object, plot all simulation elements onto it.

        :param ax: a matplotlib axis
        :param true_color: if True, the ray's actual colour used for plotting. Otherwise matplotlib handles colours
        :param ray_kwargs: keyword arguments to pass to ax.plot when drawing rays
        :return: None
        """
        for ray in self.rays:
            ax.plot(ray.history[:, 0], ray.history[:, 1], c=ray.c if true_color else None, **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)

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
        :return:
        """
        for ray in self.rays:
            ray.propagate(d)


class Ray:
    """
    A ray for the ray tracing simulation.
    """
    def __init__(self, origin, direction, wavelength=467):
        """
        Create a new ray.

        :param origin: 1D, 2-element list describing the ray's origin in 2D space (x,y)
        :param direction: A unit vector pointed in the ray's propagation direction
        """
        self._origin = np.array(origin)
        self.history = np.array([self._origin])
        self.dir = normalize(np.array(direction))
        self.wavelength=wavelength
        self.c = nm_to_rgb(wavelength)
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
        self.history = np.append(self.history, [self._origin], axis=0)

    @property
    def angle(self):
        """
        The direction of the ray described as an angle in radians, where 0 corresponds to
        the ray moving along the x axis.

        :return: a float representing the angle in radians
        """
        return np.arctan2(*self.dir[::-1])

    def __repr__(self):
        return "Ray({}, {})".format(self._origin, self.dir)


class TracerObject:
    """
    Base class for all objects that interact with rays in this ray tracer.
    """
    def __init__(self, origin, n_out=1, n_in=1):
        """
        Create a new TracerObject.

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
        ray.origin = point
        change = 2 * self.normal(point) * np.dot(self.normal(point), ray.dir)
        ray.dir -= change

        # Calculate the change in momentum
        if np.dot(ray.dir, self.normal(point)) < 0:
            n = self.n_out(ray.wavelength) if callable(self.n_out) else self.n_out
        else:
            n = self.n_in(ray.wavelength) if callable(self.n_in) else self.n_in
        self.momenta.append(change * n / ray.wavelength)
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
            m_init = ray.dir * n1 / ray.wavelength
            ray.dir = n1/n2*ray.dir + (n1/n2*cos_i - np.sqrt(1 - (n1/n2)**2 * sin_i2)) * normal
            self.momenta.append(m_init - ray.dir * n2 / ray.wavelength)
            self.m_pos.append(point)

    def act_ray(self, ray, point):
        """
        Given a ray object, interact with it (could be a reflection or refraction, or some other action).

        :param ray: a Ray object
        :param point: [X, Y] coordinates of the point of interaction
        :return: None
        """
        raise NotImplementedError

    def plot(self, ax):
        """
        Graph a representation of this object on the given matplotlib axis
        :param ax: a matplotlib axis object
        :return: None
        """
        pass


class Plane(TracerObject):
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
        raise NotImplementedError

    def plot(self, ax):
        if self.radius is None:
            points = np.array([self.origin + self.along, self.origin + 0.1 * self.along, self.origin + self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.along])
        else:
            points = np.array([self.origin + self.radius*self.along, self.origin + 0.1 * self.along, self.origin + self._normal,
                               self.origin - 0.1 * self.along, self.origin - self.radius*self.along])
        ax.plot(points[:, 0], points[:, 1], ":")


class Mirror(Plane):
    def act_ray(self, ray, point):
        if self.radius is None or np.linalg.norm(point - self.origin) <= self.radius:
            self.reflect(ray, point)
        else:
            ray.origin = point

    def __repr__(self):
        return "Mirror({}, {})".format(self.origin, self._normal)


class RefractiveSurface(Plane):
    def act_ray(self, ray, point):
        if self.radius is None or np.linalg.norm(point - self.origin) <= self.radius:
            self.refract(ray, point)
        else:
            ray.origin = point

    def __repr__(self):
        return "RefractiveSurface({}, {})".format(self.origin, self._normal)


class RayCanvas(Plane):
    """
    A transparent plane object that records data about the rays passing through it.
    Access .points, .wavelengths, and .c (colour) lists to get the data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points = []
        self.wavelengths = []
        self.c = []

    def act_ray(self, ray, point):
        ray.origin = point
        self.points.append(np.dot(self.along, point))
        self.wavelengths.append(ray.wavelength)
        self.c.append(ray.c)

    def __repr__(self):
        return "RayCanvas({}, {}): {}".format(self.origin, self._normal, self.points)


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
            self.refract(ray, point)
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


class ReflectiveSphere(Sphere):
    def act_ray(self, ray, point):
        angle = np.arctan2(*(point - self.origin)[::-1])
        if (self.mask is None or
                (self.mask[0] < self.mask[1] and self.mask[0] <= angle <= self.mask[1]) or
                (self.mask[0] > self.mask[1] and (angle >= self.mask[0] or angle <= self.mask[1]))):
            self.reflect(ray, point)
        else:
            ray.origin = point
