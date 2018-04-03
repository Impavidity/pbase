import unittest
from pbase.numericalField import NumericalField
from torchtext import data

class TestNumericalField(unittest.TestCase):
    def test_preprocess(self):
        numerical = NumericalField(batch_first=True)
        example = "1.234 9.04231 1.32454"
        ans = numerical.preprocess(example)
        expected_ans = ["1.234", "9.04231", "1.32454"]
        self.assertEqual(expected_ans, ans)

    def test_pad(self):
        numerical = NumericalField(batch_first=True)
        batch = [['1.2', '0.786', '2.11'],
                 ['1.23'],
                 ['3.1343', '5.42333', '2.5432','6.1235234']]
        ans = numerical.pad(batch)
        expected_ans = [['1.2', '0.786', '2.11', '0.0'],
                        ['1.23', '0.0', '0.0', '0.0'],
                        ['3.1343', '5.42333', '2.5432', '6.1235234']]
        self.assertEqual(expected_ans, ans)

    def test_whole_pipeline(self):
        num_field = NumericalField(batch_first=True)
        tsv_fields = [('feature', num_field)]
        tsv_dataset = data.TabularDataset(
            path="/tmp/test_numericalField.txt", format='tsv',
            fields=tsv_fields
        )
        train_iter = data.Iterator(tsv_dataset, batch_size=2, device=-1, train=True, repeat=False,
                                   sort=False, shuffle=False)
        for batch in iter(train_iter):
            print(batch.feature)

    def test_numerical_basic(self):
        num_field = NumericalField(batch_first=True)
        padded_arr = [['1.2', '0.786', '2.11', '0.0'],
                        ['1.23', '0.0', '0.0', '0.0'],
                        ['3.1343', '5.42333', '2.5432', '6.1235234']]
        tensor = num_field.numericalize(padded_arr, -1)
        print(tensor)

if __name__=="__main__":
    unittest.main()
