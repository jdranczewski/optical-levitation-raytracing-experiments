"""
This is an early version of a raytracing library, for now working only in 2 dimensions.
"""
import numpy as np
from matplotlib.patches import Circle, Wedge
from matplotlib.colors import hsv_to_rgb


def normalize(x):
    x = x/np.linalg.norm(x)
    return x


def mapping(v, a, b, c, d):
    if v < a:
        return c
    if v > b:
        return d
    else:
        return (v-a)/(b-a)*(d-c) + c


def nm_to_rgb(wvl, margin=30):
    wv = np.array([380, 460, 480, 515, 590, 630, 670])
    hv = np.array([197, 174, 135, 89, 42, 23, 0])/255
    h = np.interp(wvl, wv, hv)
    v = np.interp(wvl, [380, 380+margin, 740-margin, 740], [0,1,1,0])
    return hsv_to_rgb((h, 1, v))


class Scene:
    def __init__(self, rays, objects):
        self.rays = rays
        self.objects = objects

    def step(self):
        for ray in self.rays:
            if not ray.done:
                inter, obj = self.intersect(ray)
                if obj:
                    obj.act_ray(ray, inter)
                else:
                    ray.stop()

    def run(self, limit=100, margin=1e-10, announce_steps=False):
        for i in range(limit):
            self.step()
            self.propagate(margin)
            if announce_steps:
                print("Step", i)
            if all([ray.done for ray in self.rays]):
                break

    def plot(self, ax, true_color=True, ray_kwargs={}):
        for ray in self.rays:
            ax.plot(ray.history[:, 0], ray.history[:, 1], c=ray.c if true_color else None, **ray_kwargs)
        for obj in self.objects:
            obj.plot(ax)

    def intersect(self, ray):
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
        for ray in self.rays:
            ray.propagate(d)


class Ray:
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
        self.done = True
        self.propagate(1)

    def propagate(self, d):
        self.origin += self.dir * d

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        self._origin = np.array(new_origin)
        self.history = np.append(self.history, [self._origin], axis=0)

    @property
    def angle(self):
        return np.arctan2(*self.dir[::-1])

    def __repr__(self):
        return "Ray({}, {})".format(self._origin, self.dir)


class TracerObject:
    def __init__(self, origin, n_in=1, n_out=1):
        self.origin = origin
        self.n_in = n_in
        self.n_out = n_out

    def intersect_d(self, ray):
        raise NotImplementedError

    def normal(self, point):
        raise NotImplementedError

    def reflect(self, ray, point):
        ray.origin = point
        ray.dir -= 2 * self.normal(point) * np.dot(self.normal(point), ray.dir)

    def refract(self, ray, point):
        ray.origin = point

        # Check what direction the light's going
        n1 = self.n_in(ray.wavelength) if callable(self.n_in) else self.n_in
        n2 = self.n_out(ray.wavelength) if callable(self.n_out) else self.n_out
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
            ray.dir = n1/n2*ray.dir + (n1/n2*cos_i - np.sqrt(1 - (n1/n2)**2 * sin_i2)) * normal

    def act_ray(self, ray, point):
        raise NotImplementedError

    def plot(self, ax):
        pass


class Plane(TracerObject):
    def __init__(self, origin, normal, radius=None, *args, **kwargs):
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
    def __init__(self, origin, radius, mask=None, *args, **kwargs):
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
