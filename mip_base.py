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
        self.T, self.T_inv, self.aabb_min, self.aabb_max = self._translate(
            self.T, dx, dy, dz
        )

    def scale(self, sx: float, sy: float, sz: float):
        """Scale the volume."""
        self.T, self.T_inv, self.aabb_min, self.aabb_max = self._scale(
            self.T, sx, sy, sz
        )

    def rotate(self, ang_x: float, ang_y: float, ang_z: float):
        """Rotate the volume."""
        self.T, self.T_inv, self.aabb_min, self.aabb_max = self._rotate(
            self.T, ang_x, ang_y, ang_z
        )

    @ti.kernel
    def _translate(
        self, T1: T, dx: float, dy: float, dz: float
    ) -> Tuple[T, T, tm.vec3, tm.vec3]:
        return self._transform(T1, tm.translate(dx, dy, dz))

    @ti.kernel
    def _scale(
        self, T1: T, sx: float, sy: float, sz: float
    ) -> Tuple[T, T, tm.vec3, tm.vec3]:
        return self._transform(T1, tm.scale(sx, sy, sz))

    @ti.kernel
    def _rotate(
        self, T1: T, ang_x: float, ang_y: float, ang_z: float
    ) -> Tuple[T, T, tm.vec3, tm.vec3]:
        return self._transform(T1, tm.rotation3d(ang_x, ang_y, ang_z))

    @ti.func
    def _transform(self, T1: T, T2: T) -> Tuple[T, T, tm.vec3, tm.vec3]:
        aabb_min = apply_T(self.aabb_min, T2)
        aabb_max = apply_T(self.aabb_max, T2)
        aabb_min, aabb_max = tm.min(aabb_min, aabb_max), tm.max(aabb_min, aabb_max)
        T_new, T_inv = merge_T(T1, T2)
        return T_new, T_inv, aabb_min, aabb_max

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
        return apply_T(indices + 0.5, self.T)

    @ti.func
    def to_indices(self, p: tm.vec3) -> Indices:
        return ti.floor(apply_T(p, self.T_inv), dtype=ti.i32)


@ti.func
def apply_T(p, T1) -> tm.vec3:
    p4 = tm.vec4(*p, 1)
    transformed = T1 @ p4
    transformed /= transformed[3]
    return transformed[:3]


@ti.func
def merge_T(T1: T, T2: T) -> Tuple[T, T]:
    """Merge new transformation matrix."""
    T_new = T2 @ T1
    T_inv = tm.inverse(T_new)
    return T_new, T_inv
