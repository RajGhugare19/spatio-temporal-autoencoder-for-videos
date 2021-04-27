# spatio-temporal-autoencoder-for-videos

Currently this code is tailored specifically for [UCSDped1](http://www.svcl.ucsd.edu/projects/anomaly/dataset.html) 
1) Custom and minimal implementation of Conv LSTM
2) Custom and minimal implementatiom of Spatio temporal autoencoder
3) Custom and multi process data loader implementation for UCSDPed 1 dataset.


Steps to get this working
1) Create a folder `data`. Create two subfolders `train` and `val`
2) Create a folder `saved_models`
3) Store the downloaded dataset folder named UCSDped1 in the same directory.


References:

1) [Abnormal Event Detection in Videos using Spatiotemporal Autoencoder](https://arxiv.org/abs/1701.01546)
2) [ConvLSTM_pytorch by ndrplz](https://github.com/ndrplz/ConvLSTM_pytorch)
3) [Anomaly detection medium article by Hashem Sellat](https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29)
