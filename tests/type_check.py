import unittest
# from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import is_classifier
from msb.semi_booster import SemiBooster


class TypeTest(unittest.TestCase):
    def test(self):
        # Cannot pass so far.
        # check_estimator(SemiBooster)

        sb = SemiBooster()
        # Cannot pass so far.
        # It will run `fit` with some built-in datasets;
        #   however you have no chance to set the unlabeled data compatible with those datasets
        # check_estimator(sb)

        self.assertTrue(is_classifier(sb))
