# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the old indexed gather and current boolean-mask mesh reference bounds."""

import numpy as np

import newton


class MeshInertiaReferenceBounds:
    """CPU host preprocessing for small and large indexed sphere meshes."""

    params = [70, 700]
    param_names = ["sphere_resolution"]

    def setup(self, sphere_resolution):
        mesh = newton.Mesh.create_sphere(
            radius=0.5,
            num_latitudes=sphere_resolution,
            num_longitudes=sphere_resolution,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        self.indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1)
        self.indices = self.indices[: (len(self.indices) // 3) * 3]
        if len(self.indices) == 0:
            raise ValueError("Sphere mesh has no triangles")

        # Each timed strategy must select the same indexed geometry.
        if not np.array_equal(self._gather_bounds(), self._mask_bounds()):
            raise AssertionError("Gather and mask must produce identical reference bounds")

    def _gather_bounds(self):
        used = self.vertices[self.indices]
        return 0.5 * (used.min(axis=0) + used.max(axis=0))

    def _mask_bounds(self):
        referenced = np.zeros(len(self.vertices), dtype=bool)
        referenced[self.indices] = True
        used = self.vertices[referenced]
        return 0.5 * (used.min(axis=0) + used.max(axis=0))

    def time_gather(self, sphere_resolution):
        return self._gather_bounds()

    def time_mask(self, sphere_resolution):
        return self._mask_bounds()
