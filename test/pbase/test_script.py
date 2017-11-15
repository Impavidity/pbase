import unittest
from pbase import script

class TestScript(unittest.TestCase):
    def test_EmbeddingFilter(self):
        embeddingFilter = script.EmbeddingFilter("/data/GloVe/glove.840B.300d.txt")
        embeddingFilter.addCorpus("entity_test","/u1/p8shi/pycharm/QA_related_subtasks/ReverseLinking/entity.test", 1)
        embeddingFilter.setTargetEmbeddingPath("entity_test_embed.vec")
        embeddingFilter.setTargetBinaryPath("entity.pt")
        embeddingFilter.toBinary()
        embeddingFilter.extractAverageEmbedding("entity_test")
    def test_LinguisticFeatureAnnotator(self):
        annotator = script.LinguisticFeatureAnnotator("tools")
        annotator.addCorpus('test', '/u1/p8shi/pycharm/QA_related_subtasks/JointPrediction/data/annotated_fb_entity_test.txt', 0)
        annotator.annotate('test')


if __name__=="__main__":
    unittest.main()

