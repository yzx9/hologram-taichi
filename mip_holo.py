import time

import nrrd
import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda, device_memory_GB=4)

from mip_base import Camera, Ray, Volume

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
wavelength = 670 * nm  # wavelength of the red light


@ti.func
def get_complex_amplitude(
    ray: Ray, p: tm.vec3, p_amp: float, p_phase: float
) -> tm.vec2:
    # the distance between the scene point and the hologram point
    dis = tm.distance(ray.origin, p)

    # print(scene_point, hologram_point)
    # print(r))
    complex_amplitude = (
        p_amp * tm.exp(tm.vec2(0, 1) * (2 * tm.pi / wavelength * dis + p_phase)) / dis
    )

    return complex_amplitude


@ti.data_oriented
class Renderer:
    def __init__(self, src: str) -> None:
        data, _ = nrrd.read(src)
        data = data / 255
        print("data", data.shape, "max", data.max())
        field = ti.field(ti.f32, shape=data.shape)
        field.from_numpy(data)
        self.volume = Volume(field, tm.vec3(-5, -5, 1), tm.vec3(10, 10, 0.2))

        self.width, self.height = 640, 480
        self.image = ti.field(ti.f32, shape=(self.width, self.height))
        self.hologram = ti.field(tm.vec2, shape=(self.width, self.height))

        # generate a random phase
        self.phase = ti.field(ti.f32, shape=data.shape)

        @ti.kernel
        def random_phase():
            for i, j, k in self.phase:
                self.phase[i, j, k] = ti.random() * 2 * tm.pi - tm.pi

        random_phase()

    @ti.kernel
    def update_image(self):
        for i, j in self.image:
            origin = tm.vec3((ti.f32(i) - 320) / 40, (ti.f32(j) - 240) / 40, 0)
            direction = tm.vec3(0, 0, 1)
            up = tm.vec3(0, 1, 0)
            aspect_ratio = self.width / self.height
            camera = Camera(origin, direction, up, 60, aspect_ratio)

            self.hologram[i, j] = ti.zero(self.hologram[0, 0])
            N = 120
            for uv in range(N * N):
                u = uv // N
                v = uv % N

                # calcute the ray
                x = (u + 0.5) / N
                y = (v + 0.5) / N
                ray = camera.generate_ray(x, y)

                # cast the ray
                amp, p, hit = self.volume.cast_mip(ray)
                if hit:
                    phase = self.phase[p]
                    self.hologram[i, j] += get_complex_amplitude(ray, p, amp, phase)

                if u == N / 2 and v == N / 2:
                    if hit:
                        self.image[i, j] = amp
                    else:
                        self.image[i, j] = ti.zero(amp)

    def run(self):
        t0 = time.time()
        self.update_image()
        print("time", time.time() - t0)

        gui = ti.GUI("Window Title", (self.width, self.height))
        while gui.running:
            gui.set_image(self.image)
            gui.show()

        gui.show("tmp.png")
        hologram = self.hologram.to_numpy(np.float32)
        print(hologram.shape)
        nrrd.write("tmp_hologram.nrrd", hologram)

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
