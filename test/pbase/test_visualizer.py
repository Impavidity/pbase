import unittest
from pbase import visualizer

class TestScript(unittest.TestCase):
    def test_distribution(self):
        dis_template = visualizer.KeyInFileDistribution('/u1/p8shi/pycharm/QA_related_subtasks/Linking/todo.test', keyIndex=1)
        dis_template.sorted_cumulative_plot(xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_template.sorted_cumulative_plot(k=1000, xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_template.sorted_cumulative_plot(k=200, xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_relation = visualizer.KeyInFileDistribution('/u1/p8shi/pycharm/QA_related_subtasks/Linking/todo.test', keyIndex=2)
        dis_relation.topk(20)

    def test_comparison(self):
        comparison = visualizer.DistributionComparison("NER", "F1", "LSTM", "GRU", [80.5, 80.6, 80.7, 80.8],
                                                       [80.7, 80.8, 80.9, 81.0])
        comparison.draw_violin_plot()
        comparison.ks_significance()
        comparison.levene_significance()

if __name__=="__main__":
    unittest.main()