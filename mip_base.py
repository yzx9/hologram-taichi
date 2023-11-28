from typing import Tuple

import nrrd
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
                ret = ti.atomic_min(ret, self.data[i, j, k])

            return ret

        self.data_min = data_min()

    @ti.func
    def sample(self, point: tm.vec3) -> Tuple[float, bool]:
        coord = ti.round((point - self.origin) / self.grid_size, dtype=ti.i32)
        value = ti.zero(self.data[0, 0, 0])
        hit = False
        if (
            0 <= coord[0] < self.data.shape[0]
            and 0 <= coord[1] < self.data.shape[1]
            and 0 <= coord[2] < self.data.shape[2]
        ):
            value = self.data[coord[0], coord[1], coord[2]]
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

        pmax = ray.cast(t0)
        vmax = self.data_min
        hit = False
        if t0 <= t1:
            # there is intersection between the ray and the AABB
            step = 0.002
            hit = True

            for i in range(1000):
                t = t0 + i * step
                if t > t1:
                    break

                p = ray.cast(t)
                v, ok = self.sample(p)
                if ok and v > vmax:
                    pmax = p
                    vmax = v

        return vmax, pmax, hit


@ti.data_oriented
class Renderer:
    def __init__(self, src: str) -> None:
        data, _ = nrrd.read(src)
        self.field = ti.field(ti.f32, shape=data.shape)
        self.field.from_numpy(data / 256)

        self.volume = Volume(self.field, tm.vec3(-5, -5, 1), tm.vec3(10, 10, 0.2))

        self.width, self.height = 640, 480
        aspect_ratio = self.width / self.height
        origin = tm.vec3(0, 0, 0)
        direction = tm.vec3(0, 0, 1)
        up = tm.vec3(0, 1, 0)
        self.camera = Camera(origin, direction, up, 60, aspect_ratio)
        self.image = ti.field(ti.f32, shape=(self.width, self.height))

    @ti.kernel
    def update_image(self):
        for i, j in self.image:
            # calcute the ray
            x = (i + 0.5) / self.width
            y = (j + 0.5) / self.height
            ray = self.camera.generate_ray(x, y)

            # cast the ray
            value, point, hit = self.volume.cast_mip(ray)
            # print(value, point, hit)
            if hit:
                self.image[i, j] = value
            else:
                self.image[i, j] = ti.zero(value)

    def run(self):
        gui = ti.GUI("Window Title", (self.width, self.height))
        while gui.running:
            self.update_image()
            gui.set_image(self.image)
            gui.show()

    @ti.kernel
    def test_run(self):
        # calcute the ray
        ray = Ray(tm.vec3(0, 0, 0), tm.vec3(0, 0, 1))

        # cast the ray
        value, point, hit = self.volume.cast_mip(ray)
        print("cast", value, point, hit)
