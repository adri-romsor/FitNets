FitNets
=======

FitNets: Hints for Thin Deep Nets

http://arxiv.org/abs/1412.6550

- To run FitNets stage-wise training:
  THEANO_FLAGS="device=gpu,floatX=float32,optimizer_including=cudnn" python fitnets_training.py fitnet_yaml regressor,
  where fitnet_yaml is the path to the FitNet yaml file and regressor is the regressor type, either convolutional (conv) or   fully-connected (fc). 

- yaml files to reproduce the experiments will be available soon
