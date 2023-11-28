from typing import Tuple

import taichi as ti
import taichi.math as tm


@ti.dataclass
class Ray:
    origin: tm.vec3
    direction: tm.vec3

    def __init__(self, origin: tm.vec3, direction: tm.vec3):
        super().__init__()
        self.origin = origin
        self.direction = tm.normalize(direction)

    @ti.func
    def cast(self, t: float) -> tm.vec3:
        return self.origin + t * self.direction


@ti.dataclass
class Camera:
    origin: tm.vec3
    direction: tm.vec3
    up: tm.vec3
    fov: float
    aspect_ratio: float

    def __init__(
        self,
        origin: tm.vec3,
        direction: tm.vec3,
        up: tm.vec3,
        fov: float,
        aspect_ratio: float,
    ):
        super().__init__()
        self.origin = origin
        self.direction = tm.normalize(direction)
        self.up = tm.normalize(up)
        self.fov = fov
        self.aspect_ratio = aspect_ratio

    @ti.func
    def generate_ray(self, x: float, y: float) -> Ray:
        """generate a ray from the camera to the (x, y) point in the image

        Parameters
        ----------
        x: the x coordinate of the point in the image
        y: the y coordinate of the point in the image

        Returns
        -------
        ray: the ray from the camera to the point
        """

        x = 2 * tm.clamp(x, 0, 1) - 1
        y = 2 * tm.clamp(y, 0, 1) - 1

        # Calculate the direction of the ray
        # TODO: direction and up
        w = tm.tan(self.fov / 2)
        h = w / self.aspect_ratio
        direction = tm.normalize(tm.vec3(x * w, y * h, 1))

        return Ray(self.origin, direction)


@ti.data_oriented
class Volume:
    data: ti.Field  # the data of the volume
    origin: tm.vec3
    size: tm.vec3

    grid_size: tm.vec3
    data_min: float

    def __init__(self, data: ti.Field, origin: tm.vec3, size: tm.vec3):
        super().__init__()

        assert len(data.shape) == 3
        self.data = data
        self.origin = origin
        self.size = size

        self.grid_size = self.size / self.data.shape

        @ti.kernel
        def data_min() -> ti.f32:
            ret: ti.f32 = self.data[0, 0, 0]

            for i, j, k in self.data:
                ti.atomic_min(ret, self.data[i, j, k])

            return ret

        self.data_min = data_min()

    @ti.func
    def sample(self, p) -> Tuple[float, bool]:
        value = ti.zero(self.data[0, 0, 0])
        hit = False
        if (
            0 <= p[0] < self.data.shape[0]
            and 0 <= p[1] < self.data.shape[1]
            and 0 <= p[2] < self.data.shape[2]
        ):
            value = self.data[p]
            hit = True

        return value, hit

    @ti.func
    def cast_mip(self, ray: Ray) -> Tuple[float, tm.vec3, bool]:  # value, point, hit
        # calcute the AABB of the volume
        aabb_min = self.origin
        aabb_max = self.origin + self.size

        # calcute the intersection of the ray and the AABB
        ta = (aabb_min - ray.origin) / ray.direction
        tb = (aabb_max - ray.origin) / ray.direction

        tmin = tm.min(ta, tb)
        tmax = tm.max(ta, tb)

        # the ray must be in front of the start point
        tmin = tm.max(tmin, 0.0)

        # the max enter point and the min exit point of the ray
        t0 = tm.max(*tmin)
        t1 = tm.min(*tmax)

        # print("aabb", aabb_min, aabb_max)
        # print("ray", ray.origin, ray.direction)
        # print("t", t0, t1)

        coord_max = ti.Vector([0, 0, 0], dt=ti.i32)
        vmax = self.data_min
        hit = False
        if t0 <= t1:
            # there is intersection between the ray and the AABB
            step = 0.003
            hit = True

            for i in range(100):
                t = t0 + i * step
                if t > t1:
                    break

                p = ray.cast(t)
                coord = ti.round((p - self.origin) / self.grid_size, dtype=ti.i32)
                v, ok = self.sample(coord)
                if ok and v > vmax:
                    coord_max = coord
                    vmax = v

        return vmax, coord_max, hit
