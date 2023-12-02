from typing import Tuple

import taichi as ti
import taichi.math as tm

T = ti.types.matrix(4, 4, ti.f32)
Indices = ti.types.vector(3, ti.i32)


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
    data_min: float

    T: T  # from the volume to the world
    T_inv: T  # from the world to the volume

    aabb_min: tm.vec3
    aabb_max: tm.vec3

    def __init__(self, data: ti.Field, origin: tm.vec3, size: tm.vec3):
        super().__init__()

        assert len(data.shape) == 3
        self.data = data

        self.T = ti.Matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dt=ti.f32
        )
        self.T_inv = ti.Matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dt=ti.f32
        )
        self.aabb_min = tm.vec3(0, 0, 0)
        self.aabb_max = tm.vec3(data.shape[0], data.shape[1], data.shape[2])

        scaler = size / self.data.shape
        self.scale(scaler[0], scaler[1], scaler[2])

        self.translate(origin[0], origin[1], origin[2])

        @ti.kernel
        def data_min() -> ti.f32:
            ret: ti.f32 = self.data[0, 0, 0]

            for i, j, k in data:
                ti.atomic_min(ret, self.data[i, j, k])

            return ret

        self.data_min = data_min()

    # Gerometry Transformation

    def translate(self, dx: float, dy: float, dz: float):
        """Translate the volume."""
        self.T, self.T_inv = translate(self.T, dx, dy, dz)
        self._update_aabb()

    def scale(self, sx: float, sy: float, sz: float):
        """Scale the volume."""
        self.T, self.T_inv = scale(self.T, sx, sy, sz)
        self._update_aabb()

    def rotate(self, ang_x: float, ang_y: float, ang_z: float):
        """Rotate the volume."""
        self.T, self.T_inv = rotate(self.T, ang_x, ang_y, ang_z)
        self._update_aabb()

    def _update_aabb(self):
        @ti.kernel
        def aabb() -> Tuple[tm.vec3, tm.vec3]:
            shape = self.data.shape
            aabb_min = self.to_world(ti.Matrix([0, 0, 0]))
            aabb_max = self.to_world(ti.Matrix([shape[0], shape[1], shape[2]]))
            aabb_min, aabb_max = tm.min(aabb_min, aabb_max), tm.max(aabb_min, aabb_max)
            return aabb_min, aabb_max

        self.aabb_min, self.aabb_max = aabb()

    # Ray Casting

    @ti.func
    def sample(self, p):
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
    def cast_mip(self, ray: Ray):  # value, indices, hit
        # calcute the AABB of the volume
        aabb_min = self.aabb_min
        aabb_max = self.aabb_max

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

        indices_max = ti.Vector([0, 0, 0], dt=ti.i32)
        vmax = self.data_min
        hit = False
        if t0 <= t1:
            # there is intersection between the ray and the AABB
            step = 0.003
            hit = True

            for i in range(1000):
                t = t0 + i * step
                if t > t1:
                    break

                p = ray.cast(t)
                indices = self.to_indices(p)
                v, ok = self.sample(indices)
                if ok and v > vmax:
                    indices_max = indices
                    vmax = v

        return vmax, indices_max, hit

    @ti.func
    def to_world(self, indices: Indices) -> tm.vec3:
        pp = tm.vec4(*indices, 1)
        p = self.T @ pp
        p /= p[3]
        return p[:3]

    @ti.func
    def to_indices(self, p: tm.vec3) -> Indices:
        pp = tm.vec4(*p, 1)
        local = self.T_inv @ pp
        local /= local[3]
        indices = ti.round(local, dtype=ti.i32)
        return indices[:3]


@ti.kernel
def translate(T1: T, dx: float, dy: float, dz: float) -> Tuple[T, T]:
    T_new = tm.translate(dx, dy, dz)
    return merge_transformation(T1, T_new)


@ti.kernel
def scale(T1: T, sx: float, sy: float, sz: float) -> Tuple[T, T]:
    T_new = tm.scale(sx, sy, sz)
    return merge_transformation(T1, T_new)


@ti.kernel
def rotate(T1: T, ang_x: float, ang_y: float, ang_z: float) -> Tuple[T, T]:
    T_new = tm.rotation3d(ang_x, ang_y, ang_z)
    return merge_transformation(T1, T_new)


@ti.func
def merge_transformation(T1: T, T2: T) -> Tuple[T, T]:
    """Merge new transformation matrix."""
    T_new = T2 @ T1
    T_inv = tm.inverse(T_new)
    return T_new, T_inv
