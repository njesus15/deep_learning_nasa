# deep_learning_nasa
The 'core' directory contains a variety of files that each serve different purposes. 
There are three main implementations: 
### 1. cnn_model.py -
  Reimplementation of the convolutional neural network trained to classify time-series images
  following a forest trail as facing 'right', 'center' or 'left'. The [dataset](http://people.idsia.ch/~giusti/forest/web/)   
  contains over 20,000 images.
### 2. knn.py -
  Trains a knn classifier using the output of a specified representational layer of a defined neural network model.
  The idea was to explore the transfer learning of two models (cnn (1) and VGG16) on forest trail images collected at different locations. Using a knn classifier, the goal was to use a distance metric to reason about how out-of distribution data populates the model's representational layer output on 2 cases: (i) The knn fit is initially trained on in-distribution data, and (2) the classifier has no reference and instead is initially populated with a subset of the out-of-distribution dataset. 

### 3. deepknn.py -
  INCOMPLETE implementation [of deep k-nearest neighbors](https://arxiv.org/pdf/1803.04765.pdf)
  
### 4. dataset.py -
  Processing of data 
  
# Results
Have not been pushed to repo. 
