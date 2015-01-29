 
import sys, getopt
import os.path as op

# pylearn2 imports
from pylearn2.config import yaml_parse

# my imports
import FitNets.scripts.fitnets_training as ft
#import FitNets.scripts.make_yamls as my
import FitNets.scripts.make_fitnets_yamls as my

fitnet_save_path = '/data/lisatmp2/romerosa/evol_boost'

def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    fitnet_yaml_template = args[0]
    teacher_path = args[1]
    regressor_type = args[2]
    nber_fitnets = int(args[3])
  except getopt.GetoptError:
    usage()
    sys.exit(2) 
      

  for i in range(nber_fitnets):
    print 'Evolutionary Boosting: FitNet %d out of %d' % (i+1, nber_fitnets)
    
    path, filename = op.split(fitnet_yaml_template)
    fitnet_name = filename[0:-5] + '_' + str(i)
    fitnet_yaml = op.join(path,fitnet_name + '.yaml')

    print '...Preparing yaml file'
    my.main([fitnet_yaml_template,fitnet_name,fitnet_save_path,teacher_path])
    
    print '...Training FitNet'
    ft.execute(fitnet_yaml, regressor_type)
    
    teacher_path = op.join(fitnet_save_path,fitnet_name + '_best.pkl')    
  
if __name__ == "__main__":
  main(sys.argv[1:])
