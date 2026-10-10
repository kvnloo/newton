"""Audit representability at the public shape-transform boundary."""
import math
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton._src.geometry.raycast import map_ray_to_local

@wp.kernel
def map_probe(transform:wp.transform,origins:wp.array[wp.vec3],directions:wp.array[wp.vec3],out_o:wp.array[wp.vec3],out_d:wp.array[wp.vec3]):
 i=wp.tid()
 o,d=map_ray_to_local(transform,origins[i],directions[i])
 out_o[i]=o
 out_d[i]=d

wp.config.kernel_cache_dir=str(Path(__file__).parent/'warp-cache')
wp.init()
x_in=np.float32(.5-2.0**-22)
x_out=np.float32(.5+2.0**-22)
o=np.array([[x_in,0,0],[x_out,0,0],[x_in,0,0]],dtype=np.float32)
d=np.array([[-1,0,2],[-1,0,2],[-1,0,4*x_in]],dtype=np.float64)
d/=np.linalg.norm(d,axis=1)[:,None]
d=d.astype(np.float32)
wo=o[:,[2,0,1]]+np.array([3,-2,5],dtype=np.float32)
wd=d[:,[2,0,1]]
xform=wp.transform(wp.vec3(3,-2,5),wp.quat(.5,.5,.5,.5))
builder=newton.ModelBuilder()
builder.begin_world()
builder.add_shape_cone(body=-1,xform=xform,radius=1.0,half_height=1.0)
builder.end_world()
model=builder.finalize(device='cpu')
model_transform=model.bvh_shape_world_transforms.numpy()[0]
print('actual BVH transform',model_transform)
assert np.array_equal(model_transform,np.array([3,-2,5,.5,.5,.5,.5],dtype=np.float32))
xform=wp.transform(wp.vec3(*model_transform[:3]),wp.quat(*model_transform[3:]))
out_o=wp.empty(3,dtype=wp.vec3,device='cpu')
out_d=wp.empty(3,dtype=wp.vec3,device='cpu')
wp.launch(map_probe,dim=3,inputs=[xform,wp.array(wo,dtype=wp.vec3,device='cpu'),wp.array(wd,dtype=wp.vec3,device='cpu')],outputs=[out_o,out_d],device='cpu')
np.set_printoptions(precision=12)
print('intended local origins',o)
print('actual world origins',wo)
print('mapped local origins',out_o.numpy())
print('intended local directions',d)
print('actual mapped local directions',out_d.numpy())
print('origin bits intended',o.view(np.uint32))
print('origin bits mapped',out_o.numpy().view(np.uint32))
