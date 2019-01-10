class MetricsComparator(object):

  def __init__(self, index):
    self.index = index

  def __call__(self, new_metrics, best_metrics):
    if best_metrics == None or new_metrics[self.index] >= best_metrics[self.index]:
      return True
    return False
