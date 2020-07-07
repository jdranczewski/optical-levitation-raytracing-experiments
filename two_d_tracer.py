"""
This is an early version of a raytracing library, for now working only in 2 dimensions.
"""
import numpy as np


def normalize(x):
    x = x/np.linalg.norm(x)
    return x


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
    def __init__(self, origin, direction):
        """
        Create a new ray.

        :param origin: 1D, 2-element list describing the ray's origin in 2D space (x,y)
        :param direction: A unit vector pointed in the ray's propagation direction
        """
        self._origin = np.array(origin)
        self.history = np.array([self._origin])
        self.dir = normalize(np.array(direction))
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
        if np.dot(ray.dir, self._normal) < 0:
            n1 = self.n_in
            n2 = self.n_out
            normal = self._normal
        else:
            n1 = self.n_out
            n2 = self.n_in
            normal = -self._normal

        # Check for total internal reflection
        cos_i = -np.dot(ray.dir, normal)
        sin_i2 = 1 - cos_i**2
        if np.sqrt(sin_i2) > n2/n1:
            self.reflect(ray, point)
        else:
            ray.dir = n1/n2*ray.dir + (n1/n2*cos_i - np.sqrt(1 - (n1/n2)**2 * sin_i2)) * normal

    def act_ray(self, ray, point):
        raise NotImplementedError


class Plane(TracerObject):
    def __init__(self, origin, normal, *args, **kwargs):
        super().__init__(origin, *args, **kwargs)
        self._normal = normalize(np.array(normal))

    def intersect_d(self, ray):
        dot = np.dot(self.origin - ray.origin, self._normal)
        if dot != 0:
            d = dot / np.dot(ray.dir, self._normal)
            return d if d > 0 else np.inf
        else:
            return np.inf

    def normal(self, point):
        return self._normal

    def act_ray(self, ray, point):
        raise NotImplementedError


class Mirror(Plane):
    def act_ray(self, ray, point):
        self.reflect(ray, point)

    def __repr__(self):
        return "Mirror({}, {})".format(self.origin, self._normal)


class RefractiveSurface(Plane):
    def act_ray(self, ray, point):
        self.refract(ray, point)

    def __repr__(self):
        return "Mirror({}, {})".format(self.origin, self._normal)