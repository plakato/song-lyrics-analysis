import rhyme_scheme_detector as rsd
import unittest


class TestRhymeDetector(unittest.TestCase):
    def test_none(self):
        result = {'perfect': None,
                 'identity': False,
                 'impefect': False,
                 'weak': False,
                 'forced': False,
                 'syllabic': False,
                 'assosnance': 0,
                 'consonance': 0}
        self.assertEqual((False, []), rsd.rhymes('That is not me', 'I was swimming someplace else'))

    def test_perfect_masc(self):
        result = {'perfect': 'masculine',
                 'identity': False,
                 'impefect': False,
                 'weak': False,
                 'forced': False,
                 'syllabic': False,
                 'assosnance': 0,
                 'consonance': 0}
        self.assertEqual((True, result), rsd.rhymes('Oh, what a rhyme', 'it\'s so sublime'))

    def test_perfect_fem(self):
        result = {'perfect': 'feminine',
                 'identity': False,
                 'impefect': False,
                 'weak': False,
                 'forced': False,
                 'syllabic': False,
                 'assosnance': 0,
                 'consonance': 0}
        self.assertEqual((True, result), rsd.rhymes('You are so picky', 'making it tricky'))

    def test_perfect_fem_multisyll(self):
        result = {'perfect': 'feminine',
                 'identity': False,
                 'impefect': False,
                 'weak': False,
                 'forced': False,
                 'syllabic': False,
                 'assosnance': 0,
                 'consonance': 0}
        self.assertEqual((True, result), rsd.rhymes('I said you\'re a poet', 'it\'s true and you know it'))

    def test_perfect_dact(self):
        result = {'perfect': 'dactylic',
                 'identity': False,
                 'impefect': False,
                 'weak': False,
                 'forced': False,
                 'syllabic': False,
                 'assosnance': 0,
                 'consonance': 0}
        self.assertEqual((True, result), rsd.rhymes('Sometimes I\'m amorous', 'but always glamorous'))


if __name__ == '__main__':
    unittest.main()
