class Example(object):
  """Define the example class
  """

  def __init__(self, example_dict, attributes):
    self.example = example_dict
    self.attributes = attributes
    for attribute in attributes:
      setattr(self,
              attribute.target,
              attribute.field.preprocess(example_dict[attribute.source]))

  def __str__(self):
    return ("Example:\n"
     "Raw Data:\n" +
      str(self.example) +
      "\nProcessed Data:\n" +
      "\n".join([attribute.target + ":\n" + str(getattr(self, attribute.target))
                 for attribute in self.attributes]))