"""Check frozen rest-state invariant and independent physical orientation on CPU."""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import warp as wp
import newton

parser = argparse.ArgumentParser()
parser.add_argument('--expected-root', required=True)
args = parser.parse_args()
root = pathlib.Path(args.expected_root).resolve()
assert pathlib.Path(newton.__file__).resolve() == root / 'newton/__init__.py'
print('IMPORT_OK', root.name, 'WARP', wp.__version__, 'NUMPY', np.__version__)
ATOL = 1e-8
POSE_ATOL = 1e-6
AXES = [(1.,0.,0.), (0.,1.,0.), (0.,0.,1.)]
PAIRS = [(0,1), (1,0), (1,2), (2,1), (2,0), (0,2)]
CASES = [(f'{i}/{j}', AXES[i], AXES[j]) for i,j in PAIRS]
CASES += [('rotated', (2**-0.5,2**-0.5,0.), (0.,0.,1.))]

def quaternion(axis, angle):
    return np.r_[np.asarray(axis)*np.sin(angle/2), np.cos(angle/2)]

def mul(a,b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]), a[3]*b[3]-np.dot(a[:3],b[:3])]

def run(name, axis0, axis1, framed=False, offset=0.):
    axes = np.asarray([axis0,axis1]); np.testing.assert_allclose(axes@axes.T,np.eye(2),atol=1e-12,rtol=0.)
    q = (.2,.4)
    # Nontrivial anchors are independent of the two joint axes.
    p = quaternion((0.,0.,1.),.31) if framed else np.array([0.,0.,0.,1.])
    c = quaternion((1.,0.,0.),-.27) if framed else np.array([0.,0.,0.,1.])
    builder = newton.ModelBuilder(gravity=(0.,0.,0.))
    body = builder.add_link(mass=1.,inertia=wp.mat33(np.eye(3)),lock_inertia=True)
    joint = builder.add_joint_d6(-1,body,
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis0,target_pos=q[0]+offset,target_ke=2.), newton.ModelBuilder.JointDofConfig(axis=axis1,target_pos=q[1],target_ke=2.)],
        parent_xform=wp.transform(wp.vec3(),wp.quat(*p)),child_xform=wp.transform(wp.vec3(),wp.quat(*c)))
    builder.add_articulation([joint]); builder.joint_q[:]=q
    model=builder.finalize(device='cpu')
    np.testing.assert_allclose(model.joint_axis.numpy(),axes,atol=1e-7,rtol=0.)
    control=model.control()
    np.testing.assert_allclose(control.joint_target_q.numpy(),[q[0]+offset,q[1]],atol=1e-7,rtol=0.)
    solver=newton.solvers.SolverSemiImplicit(model,angular_damping=0.,joint_attach_ke=0.,joint_attach_kd=0.)
    state,result=model.state(),model.state()
    newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    start_pose=state.body_q.numpy().copy()
    ci=c.copy();ci[:3]*=-1
    # Product of axis-angle rotations, without eval_ik or quat_decompose.
    expected=mul(mul(mul(p,quaternion(axis0,q[0])),quaternion(axis1,q[1])),ci)
    actual=start_pose[0,3:]
    orientation_error=min(np.max(np.abs(actual-expected)),np.max(np.abs(actual+expected)))
    assert orientation_error<POSE_ATOL, (name,orientation_error)
    np.testing.assert_allclose(state.body_qd.numpy(),0.,atol=ATOL,rtol=0.)
    state.clear_forces();solver.step(state,result,control,None,.001)
    newton.eval_ik(model,result,result.joint_q,result.joint_qd)
    vel=result.joint_qd.numpy();physical=result.body_qd.numpy();drift=np.max(np.abs(result.body_q.numpy()-start_pose))
    passed = (np.max(np.abs(vel))<ATOL and np.max(np.abs(physical))<ATOL and drift<POSE_ATOL) if offset==0. else (vel[0]>1e-6 and np.max(np.abs(physical))>1e-6)
    print(json.dumps(dict(case=name,framed=framed,offset=offset,orientation_error=float(orientation_error),qd=vel.tolist(),body_speed=float(np.max(np.abs(physical))),pose_drift=float(drift),passed=bool(passed))))
    return bool(passed)

ok=[]
for framed in (False,True):
    for name,a,b in CASES:
        ok.append(run(name,a,b,framed))
# A live drive must respond when its first target is above the rest coordinate.
ok.append(run('off_target_XY',*AXES[:2],False,.05))
print('SUMMARY',sum(ok),'PASS',len(ok)-sum(ok),'FAIL', 'ATOL',ATOL,'POSE_ATOL',POSE_ATOL)
sys.exit(0 if all(ok) else 1)
