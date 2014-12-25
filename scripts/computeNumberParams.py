import os
import sys
import getopt

from pylearn2.config import yaml_parse
from pylearn2.models.maxout import MaxoutConvC01B, Maxout
from pylearn2.models.mlp import Softmax
from pylearn2.utils import serial

def numberParams(model):
  
  params = 0
  
  previous_output = model.model.input_space.num_channels
  
  for i in range(0,len(model.model.layers)):
    
    if isinstance(model.model.layers[i], MaxoutConvC01B):
      kernel = model.model.layers[i].kernel_shape[0]*model.model.layers[i].kernel_shape[1]
      
      params = params + model.model.layers[i].num_pieces*kernel*previous_output*model.model.layers[i].output_space.num_channels
      previous_output = model.model.layers[i].output_space.num_channels
    elif isinstance(model.model.layers[i], Maxout):
      if isinstance(model.model.layers[i-1], MaxoutConvC01B):
	input_space = model.model.layers[i].input_space.shape[0]*model.model.layers[i].input_space.shape[1]
	params = params + model.model.layers[i].num_pieces*input_space*model.model.layers[i].input_space.num_channels*model.model.layers[i].output_space.dim
      elif isinstance(model.model.layers[i-1],Maxout):
	params = params + model.model.layers[i].num_pieces*model.model.layers[i].input_space.dim*model.model.layers[i].output_space.dim
    elif isinstance(model.model.layers[i], Softmax):
      if isinstance(model.model.layers[i-1], MaxoutConvC01B):
	input_space = model.model.layers[i].input_space.shape[0]*model.model.layers[i].input_space.shape[1]
	params = params + input_space*model.model.layers[i].input_space.num_channels*model.model.layers[i].output_space.dim
      elif isinstance(model.model.layers[i-1],Maxout):
	params = params + model.model.layers[i].input_space.dim*model.model.layers[i].output_space.dim
    else:
      print 'Unknown layer type'
          
  return params
      
def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    model_yaml = args[0]
  except getopt.GetoptError:
    usage()
    sys.exit(2) 

  # Load student
  #with open(model_yaml, "r") as sty:
    #model = yaml_parse.load(sty)
  model = serial.load_train_file(model_yaml)
        
  result = numberParams(model)
    
  print 'Number of parameters is %i' % (result)

  
if __name__ == "__main__":
  main(sys.argv[1:])
