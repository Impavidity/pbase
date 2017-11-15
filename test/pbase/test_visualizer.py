import unittest
from pbase import visualizer

class TestScript(unittest.TestCase):
    def test_distribution(self):
        dis_template = visualizer.Distribution('/u1/p8shi/pycharm/QA_related_subtasks/Linking/todo.test', keyIndex=1)
        dis_template.sorted_cumulative_plot(xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_template.sorted_cumulative_plot(k=1000, xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_template.sorted_cumulative_plot(k=200, xlabel='#Template Type', ylabel='Cumulative Distribution')
        dis_relation = visualizer.Distribution('/u1/p8shi/pycharm/QA_related_subtasks/Linking/todo.test', keyIndex=2)
        dis_relation.topk(20)


if __name__=="__main__":
    unittest.main()