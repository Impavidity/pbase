import torch


class BaseModel(object):
 """
 Basic structure for model
 """

 def __init__(self, config, model_definition):
  device = "cpu"
  if config.cuda:
    device = "cuda:{}".format(config.gpu)
  self.device = config.device = torch.device(device)
  self.model_definition = model_definition(config)


