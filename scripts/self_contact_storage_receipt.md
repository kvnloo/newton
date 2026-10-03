# Self-contact storage evidence

No allocator change. Fixed per-element rows stay as they are: each vertex and edge owns `pre_alloc` slots (`TriMeshCollisionInfo`, default 16 and 32), filled in BVH traversal order. Counts may exceed that capacity, and `resize_flags` records overflow. A demand-sized buffer needs a counting pass, an exclusive scan, and a second write into storage resized to `min(detected, max_alloc)`. Those buffers are allocated once, validated against `N * pre_alloc`, and consumed from CUDA-graph-capturable solver steps. That is larger than a safe patch.

Measurements below are from one run of `scripts/profile_self_contact_storage.py` on this machine. Detection was warmed up 3 times; the reported detection time and in-process CUDA used bytes are medians of 7 synchronized repeats. Index-buffer sizes are `warp.array.capacity` (bytes). Contact counts are sums over elements after the last repeat.

## Device

- GPU: NVIDIA GeForce RTX 3080 Ti
- arch: sm_86 (`device.arch` 86), 80 SMs
- `total_memory`: 12479037440 bytes (12288 MiB reported by the driver)
- Warp mempool enabled for the detection runs

Scenes are a single `ModelBuilder.add_cloth_grid` cloth (`cell` 0.02 m). Sparse keeps the flat grid and queries at radius `1e-6` m. Dense scales particle x/z by 0.02 and queries at radius 0.05 m. Triangle-vertex reverse lists are not allocated.

## Fixed capacity vs contacts

| Scene | Particles / tris / edges | Vertex pre / edge pre | Vertex slots, detected, stored, overflow elements | Edge slots, detected, stored, overflow elements | `resize_flags` | Fixed VT+EE index bytes | Bytes for stored pairs (8 B) | Bytes for detected pairs (8 B) | All result arrays (bytes) | Median detection (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_default_buffers | 1681 / 3200 / 4880 | 16 / 32 | 26896, 0, 0, 0 | 156160, 0, 0, 0 | `[0, 0, 0, 0]` | 1464448 | 0 | 0 | 1582232 | 0.4453799920156598 |
| dense_default_buffers | 1681 / 3200 / 4880 | 16 / 32 | 26896, 757920, 26896, 1681 | 156160, 3540670, 156160, 4880 | `[1, 0, 1, 0]` | 1464448 | 1464448 | 34388720 | 1582232 | 3.244554973207414 |
| sparse_large_mesh_high_capacity | 25921 / 51200 / 77120 | 256 / 256 | 6635776, 0, 0, 0 | 19742720, 0, 0, 0 | `[0, 0, 0, 0]` | 211027968 | 0 | 0 | 212881432 | 1.1880429810844362 |
| dense_small_mesh_high_capacity | 625 / 1152 / 1776 | 256 / 256 | 160000, 162144, 149680, 475 | 454656, 748910, 452550, 1728 | `[1, 0, 1, 0]` | 4917248 | 4817840 | 7288432 | 4960280 | 1.9095499883405864 |

The two default-buffer scenes allocate the same 1464448 index bytes. The flat sheet stores 0 pairs. The compressed sheet detects 757920 vertex-triangle and 3540670 edge-edge pairs, stores only the fixed slots, and sets both overflow flags. The 160-cell sheet with per-element capacity 256 stores 0 pairs and still reserves 211027968 index bytes (212881432 bytes for the full result struct).

With the mempool enabled, repeated allocations of the small struct did not move `total_memory - free_memory` (median delta 0 on 7 samples; the dense high-capacity case had one 33554432-byte step and six zeros). The large high-capacity case grew the pool by a median 201326592 bytes per extra struct (samples `[201326592, 234881024, 201326592, 201326592, 234881024, 201326592, 201326592]`), which is pool granularity, not the 212881432-byte struct. Process used bytes after detection (median of 7): sparse default 2148859904, dense default 2148859904, sparse large 2383740928, dense small 2148859904. Those figures include the rest of the CUDA context.

## Mempool disabled

A child process set `wp.config.enable_mempools_at_init = False` and allocated the same four shapes. One warmup allocation per shape was dropped. Median `total_memory - free_memory` increase over 7 retained allocations (all seven samples equal):

| Scene | Median CUDA used delta (bytes) | Declared result-struct bytes |
| --- | --- | --- |
| sparse_default_buffers | 2097152 | 1582232 |
| dense_default_buffers | 2097152 | 1582232 |
| sparse_large_mesh_high_capacity | 216006656 | 212881432 |
| dense_small_mesh_high_capacity | 6291456 | 4960280 |

Driver reservations sit a few mebibytes above the declared struct size and do not depend on how many contacts the mesh will produce.

Raw samples are in `scripts/self_contact_storage_receipt.json`.
