"""Post-candidate rotational diagnostics with exact-input Decimal geometry.

Only helper and identity-transform public paths are judged. Expected values use
actual submitted float32 values, not intended doubles or lost rigid transforms.
"""
from decimal import Decimal as D, localcontext
import json
import math
from pathlib import Path
import numpy as np
import warp as wp
import newton
from cone_contract import direct_probe


def oracle(origin,direction,radius=1.0,height=1.0):
    with localcontext() as context:
        context.prec=80
        x,y,z=map(lambda v:D.from_float(float(v)),origin)
        dx,dy,dz=map(lambda v:D.from_float(float(v)),direction)
        r,h=D.from_float(float(radius)),D.from_float(float(height))
        slope=r/(2*h);s=slope*(h-z)
        a=dx*dx+dy*dy-slope*slope*dz*dz
        b=2*(x*dx+y*dy+slope*slope*(h-z)*dz)
        c=x*x+y*y-s*s
        candidates=[]
        if dz:
            t=(-h-z)/dz
            if t>=0 and (x+t*dx)**2+(y+t*dy)**2 <= r*r: candidates.append(t)
        roots=[]
        if a:
            discriminant=b*b-4*a*c
            if discriminant>=0:
                roots=[(-b-discriminant.sqrt())/(2*a),(-b+discriminant.sqrt())/(2*a)]
        elif b:roots=[-c/b]
        elif not c:
            if -h<=z<=h:roots=[D(0)]
            elif dz:
                lo,hi=sorted([(-h-z)/dz,(h-z)/dz])
                if hi>=0:roots=[max(D(0),lo)]
        for t in roots:
            if t>=0 and -h<=z+t*dz<=h:candidates.append(t)
        return float(min(candidates)) if candidates else -1.0


cases=[]
for power in (18,22):
 for i in range(32):
    angle=2*math.pi*i/32
    co,si=math.cos(angle),math.sin(angle)
    for inside in (True,False):
        rho=.5+(-1 if inside else 1)*2.0**-power
        o=np.array([rho*co,rho*si,0],dtype=np.float32)
        d=np.array([-co,-si,2],dtype=np.float64);d/=np.linalg.norm(d);d=d.astype(np.float32)
        cases.append((f'rotated_near_generator_p{power}_angle{i}_inside{inside}',o,d))
# Non-axis, ordinary well-separated input rays, including axis origins.
rng=np.random.default_rng(771109)
for i in range(128):
    o=rng.uniform(-1.75,1.75,3).astype(np.float32)
    if i%8==0:o[:]=0
    d=rng.normal(size=3);d/=np.linalg.norm(d);d=d.astype(np.float32)
    cases.append((f'general_{i}',o,d))
wp.config.kernel_cache_dir=str(Path(__file__).parent/'warp-cache')
wp.init()
print('Newton',newton.__file__)
origins=wp.array(np.array([c[1] for c in cases]),dtype=wp.vec3,device='cpu')
directions=wp.array(np.array([c[2] for c in cases]),dtype=wp.vec3,device='cpu')
n=len(cases);radii=wp.ones(n,dtype=float,device='cpu');heights=wp.ones(n,dtype=float,device='cpu')
out_t=wp.empty(n,dtype=float,device='cpu');out_n=wp.empty(n,dtype=wp.vec3,device='cpu')
expecteds=[oracle(c[1],c[2]) for c in cases]
records=[]
def record(label):
    observed=out_t.numpy();failed=0
    for c,t,e in zip(cases,observed,expecteds):
        ok=math.isfinite(float(t)) and abs(float(t)-e)<=max(2e-6,abs(e)*2e-5)
        item=dict(path=label,case=c[0],origin=c[1].tolist(),direction=c[2].tolist(),expected=e,actual=float(t),passed=ok)
        records.append(item)
        if not ok:failed+=1;print('FAIL',json.dumps(item))
    print(label,f'{n-failed}/{n} passed')
wp.launch(direct_probe,dim=n,inputs=[origins,directions,radii,heights],outputs=[out_t,out_n],device='cpu')
record('helper')
builder=newton.ModelBuilder();builder.begin_world();builder.add_shape_cone(body=-1,radius=1,half_height=1);builder.end_world();model=builder.finalize(device='cpu')
worlds=wp.zeros(n,dtype=wp.int32,device='cpu')
for fast_math in (False,True):
 newton.intersect_ray(model,ray_origins=origins,ray_directions=directions,ray_worlds=worlds,out_dist=out_t,out_normal=out_n,fast_math=fast_math)
 record(f'public_fast_math={fast_math}')
output=Path(__import__('sys').argv[1])
output.write_text(json.dumps(records,indent=2)+'\n')
print('SUMMARY',sum(r['passed'] for r in records),'/',len(records))
