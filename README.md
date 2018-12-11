# RedeemTheBoar
This repository contains the code we used for the 2018 Data Science Bowl competition on kaggle (https://www.kaggle.com/c/data-science-bowl-2018). The competition required the implementation of a model that is capable of identifying a range of cell nuclei across varied conditions. 

keras_implementation is the main part of the code. It contains the image preprocessing, an implementation of a U-Net (as defined in https://arxiv.org/abs/1505.04597) using Keras. The U-Net architecture consists of a contracting path and an expansive path, each made of convolutional blocks. 

As the training data was limited in size, and skewed in content (images for which it was easier to mask the nuclei dominated the sample), data augmentation was an important part of this challenge. input_pipeline contains the data augmentation snipet keras_implementation calls during training.

Team members: Sinan Deger ([@sinandeger](https://github.com/sinandeger)), Donald Lee-Brown ([@dleebrown](https://github.com/dleebrown)), and Nesar Ramachandra ([@nesar](https://github.com/nesar)). 
