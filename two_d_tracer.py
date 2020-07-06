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
                    obj.refract(ray, inter)
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

    def __repr__(self):
        return "Ray({}, {})".format(self._origin, self.dir)


class TracerObject:
    def __init__(self):
        self.origin = None

    def intersect_d(self, ray):
        raise NotImplementedError

    def normal(self, point):
        raise NotImplementedError

    def normal_dir(self, point):
        return np.arctan2(*self.normal(point))

    def refract(self, point):
        raise NotImplementedError


class Mirror(TracerObject):
    def __init__(self, origin, normal):
        self.origin = np.array(origin)
        self._normal = normalize(np.array(normal))

    def intersect_d(self, ray):
        dot = np.dot(self.origin - ray.origin, self._normal)
        if dot < 0:
            d = dot / np.dot(ray.dir, self._normal)
            return d if d > 0 else np.inf
        else:
            return np.inf

    def normal(self, point):
        return self._normal

    def refract(self, ray, point):
        ray.origin = point
        ray.dir -= 2*self._normal*np.dot(self._normal, ray.dir)

    def __repr__(self):
        return "Mirror({}, {})".format(self.origin, self._normal)