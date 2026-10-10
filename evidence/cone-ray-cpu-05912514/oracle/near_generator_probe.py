"""Candidate-discovered diagnostics; does not alter the frozen oracle."""
import math
import sys
import numpy as np
import cone_contract as probe
x_in = float(np.float32(0.5) - np.float32(2.0**-22))
x_out = float(np.float32(0.5) + np.float32(2.0**-22))
dx = float(np.float32(1/math.sqrt(5)))
near_d = np.array([-1.0, 0.0, 4.0*x_in], dtype=np.float64)
near_d /= np.linalg.norm(near_d)
near_f = near_d.astype(np.float32).astype(np.float64)
near_t = (x_in+0.5)/(0.5*near_f[2]-near_f[0])
assert near_t*near_f[2] < 1.0
probe.CASES = [
    probe.case('near_generator_nonzero_quadratic_interior', (x_in,0,0), near_d, near_t),
    probe.case('near_generator_strict_interior', (x_in,0,0),(-1,0,2),(x_in+0.5)/(2*dx)),
    probe.case('near_generator_strict_exterior_miss', (x_out,0,0),(-1,0,2),-1,(0,0,0)),
]
# Distances use actual float32 input directions; skip frozen double witness
# self-validation, which checks the mathematically exact normalized direction.
probe.validate_witnesses = lambda: None
probe.main()
