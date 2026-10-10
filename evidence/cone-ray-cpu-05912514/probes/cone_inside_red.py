import warp as wp
from newton._src.geometry.raycast import ray_intersect_cone

@wp.kernel
def probe(t_out: wp.array(dtype=float), n_out: wp.array(dtype=wp.vec3)):
    t, n = ray_intersect_cone(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), 1.0, 1.0)
    t_out[0] = t
    n_out[0] = n

wp.init()
t_out = wp.empty(1, dtype=float, device='cpu')
n_out = wp.empty(1, dtype=wp.vec3, device='cpu')
wp.launch(probe, dim=1, inputs=[t_out, n_out], device='cpu')
print('inside cone exit ray:', t_out.numpy()[0], n_out.numpy()[0])
assert abs(t_out.numpy()[0] - 0.5) < 1e-6
