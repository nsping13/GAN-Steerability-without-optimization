
# An official pytorch implementaiton of [GAN "Steerability" without optimization](https://arxiv.org/pdf/2012.05328.pdf) 


![logo](teaser.jpg)


Generally, our methods are coded in the file BigGAN.py. Each path can be extracted by directing the "method" flag accordingly.
For easier reproducing, you can use main.py.

## Closed form solution 

![Closed form solution ](https://github.com/nsping13/GAN-Steerability-without-optimization/blob/main/User%20Specified.jpg)

Method | Flag
------------ | -------------
Linear paths | 'l_shifty', 'l_zoom' and 'l_shiftx'
Newmann paths |  'nl_shifty', 'nl_zoom' and 'nl_shiftx'
Great circle | 'gcircle_shifty', 'gcircle_zoom' and 'gcircle_shiftx'


## Principal directions
Linear principal directions are based on right singular vectors of the SVD on the FC weight matrix. and can be easily extracted by torch.svd() function. 
The relevant flags are: 'svd_linear', 'svd_gcircle' and 'svd_scircle' for the linear path, great, and small circle paths, accordingly.  





