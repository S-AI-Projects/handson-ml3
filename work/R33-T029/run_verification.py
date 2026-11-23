import unittest
suite = unittest.defaultTestLoader.discover("course-labs/r33-t029", "test_*.py")
res = unittest.TextTestRunner().run(suite)
exit(0 if res.wasSuccessful() else 1)
