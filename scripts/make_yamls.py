#! /usr/bin/env python

import os, sys, getopt
import os.path as op


def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    template_yaml = args[0]
    model_name = args[1]
    fitnet_save_path = args[2]
    teacher_path = args[3]
    hints = args[4]
  except getopt.GetoptError:
    usage()
    sys.exit(2)
 

  assert(op.exists(template_yaml))
  path, template_name = op.split(template_yaml)
  if len(path) == 0:
      path = os.getcwd()

  with open(template_yaml, 'r') as template_fh:
      template_data = template_fh.read()

      new_yaml_path = op.join(path, model_name) + '.yaml'	
	
      fh = open(new_yaml_path, 'w')
      fh.write(template_data % {
              'teacher_path': teacher_path,
	      'hint_fitnet': hints[0],
	      'hint_teacher': hints[1],
              'fitnet_save_path': op.join(fitnet_save_path, model_name + '.pkl'),
              'fitnet_save_path_best': op.join(fitnet_save_path, model_name + '_best.pkl'),
      })
      fh.close()

if __name__ == '__main__':
    main(sys.argv[1:])
