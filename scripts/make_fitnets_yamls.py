#! /usr/bin/env python

import os, sys, getopt
import os.path as op
import cPickle as pkl

def getYamlForMaxoutConv(hyperparams, layer):
  
    template = """
                 !obj:pylearn2.models.maxout.MaxoutConvC01B {
                 layer_name: %(name)s,
                 pad: %(pad)d,
                 tied_b: 1,
                 W_lr_scale: %(scaler)f,
                 b_lr_scale: %(scaler)f,                 
                 num_channels: %(NC)d,
                 num_pieces: %(NP)d,
                 kernel_shape: [%(KS)d, %(KS)d],
                 irange: %(range)f,
                 pool_shape: [%(Psh)d, %(Psh)d],
                 pool_stride: [%(Pst)d, %(Pst)d],
                 max_kernel_norm: %(maxNorm)f,
                 },
    
    """
    
    yamlStr = template % {'name': layer,
			  'pad': hyperparams[layer+"_pad"],
			  'scaler': hyperparams[layer+"_W_lr_scale"],
			  'scaler': hyperparams[layer+"_b_lr_scale"],
                          'NC': hyperparams[layer+"_num_channels"],
                          'NP': hyperparams[layer+"_num_pieces"],
                          'KS': hyperparams[layer+"_kernel_shape"],
                          'range': hyperparams[layer+"_irange"],
                          'Psh': hyperparams[layer+"_pool_shape"],
                          'Pst': hyperparams[layer+"_pool_stride"],
                          'maxNorm': hyperparams[layer+"_max_kernel_norm"]}

    return yamlStr
    
def getYamlForMaxout(hyperparams, layer):

    template = """
                 !obj:pylearn2.models.maxout.Maxout {
                 layer_name: %(name)s,
                 irange: %(range)f,
                 num_units: %(NU)d,
                 num_pieces: %(NP)d,
                 max_col_norm: %(maxNorm)f,
                 },
    
    """
    
    yamlStr = template % {'name': layer,
                          'range': hyperparams[layer+"_irange"],
                          'NU': hyperparams[layer+"_num_units"],
                          'NP': hyperparams[layer+"_num_pieces"],
			  'maxNorm': hyperparams[layer+"_max_col_norm"]}

    return yamlStr
 
 
def getYamlForSoftmax(hyperparams, layer):
    
    template = """
                 !obj:pylearn2.models.mlp.Softmax {
                 layer_name: %(name)s,
                 irange: %(range)f,
                 n_classes: %(NC)d,
                 max_col_norm: %(maxNorm)f,
                 },
    
    """
    
    yamlStr = template % {'name': layer,
                          'range': hyperparams[layer+"_irange"],
                          'NC': hyperparams[layer+"_n_classes"],
                          'maxNorm': hyperparams[layer+"_max_col_norm"]}

    return yamlStr

def getLayers(hyperparams):
    
    # Get the name of the pickle file to which to save the model
    savefile = getFilename(hyperparams)
    
    # Get the yaml representation of the layers of the model
    layersYaml = ""
    
    for i in np.arange(hyperparams['num_CL'])+1:
        layersYaml += getYamlForConvLayer(hyperparams, "C%i" % i )
        
    # Insert the global hyperparameters
    yamlContent = yamlTemplate % {'layers' : layersYaml,
				  'outfile_ext' : scratchdir + "/RandSearch_Results/" + savefile + "_best.pkl",
                                  'outfile' : scratchdir + "/RandSearch_Results/" + savefile + ".pkl"}
    return yamlContent
    
    
def main(argv):
  
  try:
    opts, args = getopt.getopt(argv, '')
    template_yaml = args[0]
    model_name = args[1]
    fitnet_save_path = args[2]
    teacher_path = args[3]
  except getopt.GetoptError:
    usage()
    sys.exit(2)
 

  assert(op.exists(template_yaml))
  path, template_name = op.split(template_yaml)
  if len(path) == 0:
      path = os.getcwd()
      
  # Load teacher
  fo = open(teacher_path, 'r')
  teacher = pkl.load(fo)
  fo.close()
  
  # Check the number of convolutional layers the teacher has and double it for the FitNet
  nconv = (len(teacher.layers) - 2)*2
  npool = 3
  pool_layers = range(nconv/npool-1, nconv, nconv/npool)
  layersYaml = ""
      
  # Generate Fitnet yaml
  with open(template_yaml, 'r') as template_fh:
      template_data = template_fh.read()

      new_yaml_path = op.join(path, model_name) + '.yaml'
     
      # Hyperparameter for each convolutional layer
      for l in range(0,nconv):
	j = l/2
  
	name = 'fitnet_conv%i'% l
	layerparams = {
	  name +'_pad' : 1,
	  name +'_tied_b' : teacher.layers[j].tied_b,
	  name +'_W_lr_scale' : teacher.layers[j].W_lr_scale,
	  name +'_b_lr_scale' : teacher.layers[j].b_lr_scale,
	  name +'_num_channels' : teacher.layers[j].num_channels/2,
	  name +'_num_pieces' : teacher.layers[j].num_pieces,
	  name +'_kernel_shape' : 3,
	  name +'_pool_shape' : 2 if l in pool_layers else 1,
	  name +'_pool_stride' : 2 if l in pool_layers else 1 ,
	  name +'_irange' : teacher.layers[j].irange,
	  name +'_max_kernel_norm' : teacher.layers[j].max_kernel_norm,
	}
      
	layersYaml += getYamlForMaxoutConv(layerparams, name)
      

      # Hyperparameter for the fully connected layer
      name = 'fitnet_fc1'
      layerparams = {
	name +'_num_units' : teacher.layers[-2].num_units/2,
	name +'_num_pieces' : teacher.layers[-2].num_pieces,
	name +'_irange' : teacher.layers[-2].irange,
	name +'_max_col_norm' : teacher.layers[-2].max_col_norm,
      }
      
      layersYaml += getYamlForMaxout(layerparams, name)
      
      # Hyperparameter for the softmax layer
      name = 'y'
      layerparams = {
	name +'_n_classes' : teacher.layers[-1].n_classes,
	name +'_irange' : teacher.layers[-1].irange,
	name +'_max_col_norm' : teacher.layers[-1].max_col_norm,
      }
      
      layersYaml += getYamlForSoftmax(layerparams, name)      
      
      # Hints
      hint_fitnet = nconv/2
      hint_teacher = (len(teacher.layers)-2)/2
      
      fh = open(new_yaml_path, 'w')
      fh.write(template_data % {
	      'layers': layersYaml,
              'teacher_path': teacher_path,
	      'hint_fitnet': hint_fitnet,
	      'hint_teacher': hint_teacher,
              'fitnet_save_path': op.join(fitnet_save_path, model_name + '.pkl'),
              'fitnet_save_path_best': op.join(fitnet_save_path, model_name + '_best.pkl'),
      })
      fh.close()

if __name__ == '__main__':
    main(sys.argv[1:])
