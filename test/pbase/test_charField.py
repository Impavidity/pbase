import unittest
from pbase.charField import CharField
from torchtext import data
from collections import Counter
import torch

class TestCharField(unittest.TestCase):
    def test_pad(self):
        char = CharField(sequential=True)
        batch = [["I", "love", "the", "world"],
                 ["While", "the", "world", "is", "always", "against", "me"]]
        ans = char.pad(batch)
        expected_ans = [[['I', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['l', 'o', 'v', 'e', '<pad>', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']],
                       [['W', 'h', 'i', 'l', 'e', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['i', 's', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['a', 'l', 'w', 'a', 'y', 's', '<pad>'],
                        ['a', 'g', 'a', 'i', 'n', 's', 't'],
                        ['m', 'e', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]]
        self.assertEqual(expected_ans, ans)

    def test_preprocessing(self):
        sent_field = CharField(sequential=True, preprocessing=lambda x: [list(y) for y in x])
        ans = sent_field.preprocess("I love the world")
        expected_ans = [['I'], ['l', 'o', 'v', 'e'], ['t', 'h', 'e'], ['w', 'o', 'r', 'l', 'd']]
        self.assertEqual(expected_ans, ans)


    def test_build_vocab(self):
        # set up fields
        sent_field = CharField(sequential=True)
        label_field = data.Field(sequential=False)
        # Write TSV dataset and construct a Dataset
        tsv_fields = [('sent', sent_field), ('label', label_field)]
        tsv_dataset = data.TabularDataset(
            path="/tmp/test_charField.txt", format='tsv',
            fields=tsv_fields)
        sent_field.build_vocab(tsv_dataset)
        expected_freqs = Counter(Counter({'l': 5, 'e': 5, 'a': 4, 'o': 3, 't': 3, 'h': 3, 'w': 3, 'i': 3, 's': 3,
                                          'r': 2, 'd': 2, 'I': 1, 'v': 1, 'W': 1, 'y': 1, 'g': 1, 'n': 1, 'm': 1}))
        self.assertEqual(expected_freqs, sent_field.vocab.freqs)
        expected_stoi = {'<unk>': 0, '<pad>': 1, 'e': 2, 'l': 3, 'a': 4, 'h': 5, 'i': 6, 'o': 7, 's': 8, 't': 9,
                         'w': 10, 'd': 11, 'r': 12, 'I': 13, 'W': 14, 'g': 15, 'm': 16, 'n': 17, 'v': 18, 'y': 19}
        self.assertEqual(expected_stoi, sent_field.vocab.stoi)
        expected_itos = ['<unk>', '<pad>', 'e', 'l', 'a', 'h', 'i', 'o', 's', 't',
                         'w', 'd', 'r', 'I', 'W', 'g', 'm', 'n', 'v', 'y']
        self.assertEqual(expected_itos, sent_field.vocab.itos)

    def test_numericalize_basic(self):
        # set up fields
        sent_field = CharField(sequential=True)
        label_field = data.Field(sequential=False)
        # Write TSV dataset and construct a Dataset
        tsv_fields = [('sent', sent_field), ('label', label_field)]
        tsv_dataset = data.TabularDataset(
            path="/tmp/test_charField.txt", format='tsv',
            fields=tsv_fields)
        sent_field.build_vocab(tsv_dataset)

        test_example_data = [[['I', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['l', 'o', 'v', 'e', '<pad>', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']],
                       [['W', 'h', 'i', 'l', 'e', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['i', 's', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['a', 'l', 'w', 'a', 'y', 's', '<pad>'],
                        ['a', 'g', 'a', 'i', 'n', 's', 't'],
                        ['m', 'e', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]]
        default_numericalized = sent_field.numericalize(
            test_example_data, device=-1)
        self.assertEqual(default_numericalized.size(), torch.Size([7, 2, 7]))

    def test_numericalize_batch_first(self):
        # set up fields
        sent_field = CharField(sequential=True, batch_first=True)
        label_field = data.Field(sequential=False)
        # Write TSV dataset and construct a Dataset
        tsv_fields = [('sent', sent_field), ('label', label_field)]
        tsv_dataset = data.TabularDataset(
            path="/tmp/test_charField.txt", format='tsv',
            fields=tsv_fields)
        sent_field.build_vocab(tsv_dataset)

        test_example_data = [[['I', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['l', 'o', 'v', 'e', '<pad>', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']],
                       [['W', 'h', 'i', 'l', 'e', '<pad>', '<pad>'],
                        ['t', 'h', 'e', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['w', 'o', 'r', 'l', 'd', '<pad>', '<pad>'],
                        ['i', 's', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                        ['a', 'l', 'w', 'a', 'y', 's', '<pad>'],
                        ['a', 'g', 'a', 'i', 'n', 's', 't'],
                        ['m', 'e', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']]]
        default_numericalized = sent_field.numericalize(
            test_example_data, device=-1)
        self.assertEqual(default_numericalized.size(), torch.Size([2, 7, 7]))

    def test_whole_pipeline(self):
        # set up fields
        sent_field = CharField(sequential=True, batch_first=True)
        label_field = data.Field(sequential=False)
        # Write TSV dataset and construct a Dataset
        tsv_fields = [('sent', sent_field), ('label', label_field)]
        tsv_dataset = data.TabularDataset(
            path="/tmp/test_charField.txt", format='tsv',
            fields=tsv_fields)
        sent_field.build_vocab(tsv_dataset)
        label_field.build_vocab(tsv_dataset)
        train_iter = data.Iterator(tsv_dataset, batch_size=2, device=-1, train=True, repeat=False,
                                   sort=False, shuffle=False)
        batch = next(iter(train_iter))






if __name__=="__main__":
    unittest.main()
