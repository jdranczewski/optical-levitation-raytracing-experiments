"""
This is an early version of a raytracing library, for now working only in 2 dimensions.
"""
import numpy as np


class Scene:
    def __init__(self, rays, objects):
        self.rays = rays
        self.objects = objects

    def step(self):
        for ray in self.rays:
            inter, obj = self.intersect(ray)

    def intersect(self, ray):
        d_min = np.inf
        obj_min = None
        for obj in self.objects:
            d = obj.intersect_d(ray)
            if d < d_min:
                d_min = d
                obj_min = obj
        return ray.origin + d_min*ray.dir, obj_min

class Ray:
    def __init__(self, origin, direction):
        """
        Create a new ray.

        :param origin: 1D, 2-element list describing the ray's origin in 2D space (x,y)
        :param direction: A unit vector pointed in the ray's propagation direction
        """
        self._origin = np.array(origin)
        self.history = np.array([self._origin])
        self.dir = np.array(direction)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        self._origin = np.array(new_origin)
        self.history = np.append(self.history, self._origin)

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

    def ABCD(self, point):
        raise NotImplementedError


class Mirror(TracerObject):
    def __init__(self, origin, normal):
        self.origin = np.array(origin)
        self._normal = np.array(normal)

    def intersect_d(self, ray):
        dot = np.dot(self.origin - ray.origin, self._normal)
        if dot < 0:
            return dot / np.dot(ray.dir, self._normal)
        else:
            return np.inf

    def normal(self, point):
        return self._normal

    def ABCD(self, point):
        raise NotImplementedError