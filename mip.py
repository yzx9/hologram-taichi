import math

import nrrd
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda, device_memory_GB=4, debug=True)

from mip_base import Camera, Ray, Volume

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
wavelength = 670 * nm  # wavelength of the red light


@ti.data_oriented
class Renderer:
    def __init__(self, src: str) -> None:
        data, _ = nrrd.read(src)
        field = ti.field(ti.f32, shape=data.shape)
        field.from_numpy(data / 256)

        self.volume = Volume(field, tm.vec3(-5, -5, 0), tm.vec3(10, 10, 0.2))

        self.width, self.height = 1920, 1080
        aspect_ratio = self.width / self.height
        origin = tm.vec3(0, 0, -1)
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
        self.volume.rotate(0, 0, 45 / 180 * math.pi)
        self.update_image()

        gui = ti.GUI("Window Title", (self.width, self.height))
        while gui.running:
            gui.set_image(self.image)
            gui.show()

        gui.show("tmp.png")

    @ti.kernel
    def test_run(self):
        # calcute the ray
        ray = Ray(tm.vec3(0, 0, 0), tm.vec3(0, 0, 1))

        # cast the ray
        value, point, hit = self.volume.cast_mip(ray)
        print("cast", value, point, hit)


def main():
    renderer = Renderer("./080926a.nrrd")
    renderer.run()


if __name__ == "__main__":
    main()
