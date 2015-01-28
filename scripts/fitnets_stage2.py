from pylearn2.config import yaml_parse
from pylearn2 import train

import os
import sys
import getopt
import cPickle as pkl
import argparse
import os.path as op

from pylearn2.utils import serial
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from FitNets.models.PretrainedLayerBlock import PretrainedLayerBlock

def main(argv):

  parser = argparse.ArgumentParser(
    description='Tool for training FitNets stage 2.'
  )
  parser.add_argument(
    'student_yaml',
    help='Location of the FitNet YAML file.'
  )
  parser.add_argument(
    'load_layer',
    type=int,
    default=None,
    help='Integer indicating the hint layer from which to start training.'
  )
  parser.add_argument(
    '--lr_scale',
    '-lrs',
    type=float,
    default=None,
    help='Optional. Float to scale the learning rate scaler.'
  )  

  args = parser.parse_args()
  assert(op.exists(args.student_yaml)) 
  
  execute(args.student_yaml, args.load_layer, args.lr_scale)
  
def execute(student_yaml, load_layer, lr_scale=None):

  # Load student
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)

  # Load pretrained fitnet
  hint_path = student.save_path[0:-4] + "_hintlayer" + str(load_layer) + ".pkl"
  pretrained_model = serial.load(hint_path)

  student.model.layers[0:load_layer+1] = pretrained_model.layers[0:load_layer+1]

  del pretrained_model

  if lr_scale is not None:
     for i in range(0,load_layer+1):
       if not isinstance(student.model.layers[i],PretrainedLayerBlock):
	student.model.layers[i].W_lr_scale = student.model.layers[i].W_lr_scale*lr_scale
	student.model.layers[i].b_lr_scale = student.model.layers[i].b_lr_scale*lr_scale

  student.main_loop()


if __name__ == "__main__":
  main(sys.argv[1:])
