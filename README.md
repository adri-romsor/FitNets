FitNets
=======

FitNets: Hints for Thin Deep Nets

http://arxiv.org/abs/1412.6550

- To run FitNets stage-wise training:
  THEANO_FLAGS="device=gpu,floatX=float32,optimizer_including=cudnn" python fitnets_training.py fitnet_yaml regressor -he hints_epochs -lrs lr_scale
  
  - fitnet_yaml: path to the FitNet yaml file,
  - regressor: regressor type, either convolutional (conv) or   fully-connected (fc),
  - Optional argument -he hints_epochs: int - number of epochs to train the 1st stage. It is set to None by default. Leave as None when using the validation set to determine the number of epochs. Set to X when using the whole training set.
  - Optional argument -lrs lr_scale: float - learning rate scaler to be applied to the pre-trained layers at the 2nd stage.

