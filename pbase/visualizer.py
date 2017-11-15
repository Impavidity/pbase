from collections import defaultdict
import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interative Agg backend')
    print("If you import this lib with jupyter, please use \n%matplotlib inline\nbefore importing module")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Distribution():
    def __init__(self, path, keyIndex, delimiter='\t'):
        self.count = defaultdict(int)
        fin = open(path)
        for line in fin.readlines():
            items = line.strip().split(delimiter)
            self.count[items[keyIndex]] += 1

    def topk(self, k=1):
        sort_dict = sorted(self.count.items(), key=lambda k:k[1], reverse=True)
        return sort_dict[:k]

    def sorted_cumulative_plot(self, k=None, xlabel=None, ylabel=None):
        if k is None:
            k = len(self.count.keys())
        number_per_key = sorted(self.count.values(), reverse=True)
        y = np.cumsum(number_per_key).astype("float32")
        # normalise to a percentage with specific span
        y_span = y[:k]
        y_span /= y.max()
        y_span *= 100.
        # prepend a 0 to y as zero stores have zero items
        y_span = np.hstack((0, y_span))
        # get cumulative percentage of stores
        x_span = np.arange(0, y_span.size)
        # plot
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(x_span, y_span)
        plt.show()
        return y

