"""Small geometry check for the candidate cutting direction on both jaws."""

import unittest

import trimesh

from ml.template_studio import cut_candidate


class CandidateCutTest(unittest.TestCase):
    def test_keeps_occlusal_side_and_closes_mesh(self):
        tooth = trimesh.creation.cylinder(radius=3, height=12, sections=64)
        for fdi, expected in ((16, (-6, 0)), (46, (0, 6))):
            with self.subTest(fdi=fdi):
                crown, loops = cut_candidate(tooth, fdi, 0)
                self.assertTrue(crown.is_watertight)
                self.assertEqual(len(loops), 1)
                self.assertAlmostEqual(crown.bounds[0, 2], expected[0])
                self.assertAlmostEqual(crown.bounds[1, 2], expected[1])


if __name__ == "__main__":
    unittest.main()
