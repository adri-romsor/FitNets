from pylearn2.config import yaml_parse
from pylearn2 import train

import os
import sys
import getopt
import cPickle as pkl

from pylearn2.utils import serial
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from FitNets.models.PretrainedLayerBlock import PretrainedLayerBlock

def main(argv):

  try:
    opts, args = getopt.getopt(argv, '')
    student_yaml = args[0]
    load_layer = int(args[1])

    if len(args) == 2 or (len(args) > 2 and int(args[2]) == 0):
      lr_pretrained = False
    elif int(args[2]) == 1:
      lr_pretrained = True

  except getopt.GetoptError:
    usage()
    sys.exit(2)

  # Load student
  student = serial.load_train_file(student_yaml)
  with open(student_yaml, "r") as sty:
    student = yaml_parse.load(sty)

  # Load pretrained fitnet
  hint_path = student.save_path[0:-4] + "_hintlayer" + str(load_layer) + ".pkl"
  pretrained_model = serial.load(hint_path)

  student.model.layers[0:load_layer+1] = pretrained_model.layers[0:load_layer+1]

  del pretrained_model

  if lr_pretrained:
     for i in range(0,load_layer+1):
       if not isinstance(student.model.layers[i],PretrainedLayerBlock):
	student.model.layers[i].W_lr_scale = 0.1*student.model.layers[i].W_lr_scale
	student.model.layers[i].b_lr_scale = 0.1*student.model.layers[i].b_lr_scale

  student.main_loop()


if __name__ == "__main__":
  main(sys.argv[1:])
