# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import pyglet

from newton._src.viewer.gl.opengl import RendererGL


class TestViewerGLPygletVersion(unittest.TestCase):
    def test_reject_pyglet_three_before_initializing_gl(self):
        """Reject unsupported pyglet 3 with version guidance before touching OpenGL."""
        for version in ("3.0.dev11", "3.0.0"):
            with (
                self.subTest(version=version),
                mock.patch.object(pyglet, "version", version),
                mock.patch.object(
                    RendererGL, "initialize_gl", side_effect=AssertionError("OpenGL initialization reached")
                ) as initialize_gl,
            ):
                with self.assertRaisesRegex(RuntimeError, r"ViewerGL requires pyglet >=2\.1\.6,<3; found pyglet 3\.0"):
                    RendererGL(headless=True, device="cpu")
                initialize_gl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
