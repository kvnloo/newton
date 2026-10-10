import numpy as np
import warp as wp
import newton

wp.init()
builder = newton.ModelBuilder()
builder.begin_world()
body = builder.add_body(xform=wp.transform_identity())
shape = builder.add_shape_cone(body=body, radius=1.0, half_height=1.0)
builder.end_world()
model = builder.finalize(device='cpu')
origins = wp.array(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device='cpu')
directions = wp.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device='cpu')
worlds = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device='cpu')
out_dist = wp.empty(1, dtype=float, device='cpu')
out_shape_id = wp.empty(1, dtype=wp.int32, device='cpu')
out_normal = wp.empty(1, dtype=wp.vec3, device='cpu')
newton.intersect_ray(model, ray_origins=origins, ray_directions=directions, ray_worlds=worlds, out_dist=out_dist, out_shape_id=out_shape_id, out_normal=out_normal, fast_math=False)
print('shape',shape,'distance',out_dist.numpy(),'shape ID',out_shape_id.numpy(),'normal',out_normal.numpy())
assert abs(float(out_dist.numpy()[0])-0.5)<1e-6
assert int(out_shape_id.numpy()[0])==shape
