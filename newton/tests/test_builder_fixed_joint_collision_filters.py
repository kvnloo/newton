# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton


class TestBuilderFixedJointCollisionFilters(unittest.TestCase):
    def test_collapse_refilters_reparented_shapes_cpu(self):
        """Retained joints filter their new neighbors after fixed-joint collapse."""
        for filter_parent in (True, False):
            with self.subTest(collision_filter_parent=filter_parent):
                builder = newton.ModelBuilder()
                root = builder.add_link(label="root")
                welded = builder.add_link(label="welded")
                tip = builder.add_link(label="tip")

                root_joint = builder.add_joint_free(root)
                fixed_joint = builder.add_joint_fixed(root, welded)
                tip_joint = builder.add_joint_revolute(
                    welded, tip, collision_filter_parent=filter_parent
                )
                builder.add_articulation([root_joint, fixed_joint, tip_joint])

                root_shape = builder.add_shape_sphere(root, radius=0.1)
                welded_shape = builder.add_shape_sphere(welded, radius=0.1)
                tip_shape = builder.add_shape_sphere(tip, radius=0.1)
                noncolliding_shape = builder.add_shape_sphere(
                    welded,
                    radius=0.1,
                    cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False),
                )

                new_neighbor_pair = (root_shape, tip_shape)
                self.assertNotIn(new_neighbor_pair, builder.shape_collision_filter_pairs)
                self.assertIn((root_shape, welded_shape), builder.shape_collision_filter_pairs)

                builder.collapse_fixed_joints()

                self.assertEqual(builder.shape_body[root_shape], builder.shape_body[welded_shape])
                self.assertNotEqual(builder.shape_body[root_shape], builder.shape_body[tip_shape])
                filters = builder.shape_collision_filter_pairs
                self.assertEqual(new_neighbor_pair in filters, filter_parent)
                self.assertEqual((welded_shape, tip_shape) in filters, filter_parent)
                self.assertEqual(filters.count(new_neighbor_pair), int(filter_parent))
                self.assertFalse(any(noncolliding_shape in pair for pair in filters))

                model = builder.finalize(device="cpu")
                # Three colliding shapes: one same-body pair, two cross-body pairs.
                # With parent filtering, both cross-body pairs are excluded.
                self.assertEqual(model.shape_contact_pair_count, 0 if filter_parent else 2)
