
# An official pytorch implementaiton of [GAN "Steerability" without optimization](https://arxiv.org/pdf/2012.05328.pdf) 


![logo](teaser.jpg)


Generally, our methods are coded in the BigGAN.py. Each path can be extracted by directing the "method" flag accordingly.
For easier reproducing, we prepared main.py to reproduce the paths proposed by our methods.

## Closed form solution 
The relevant code can be found in BigGAN.py under the flag options:
Linear paths: 'l_shifty', 'l_zoom' and 'l_shiftx'.
Newmann paths: 'nl_shifty', 'nl_zoom' and 'nl_shiftx'.
Great circle: 'gcircle_shifty', 'gcircle_zoom' and 'gcircle_shiftx'

## Principal directions
Linear principal directions are based on right singular vectors of the SVD on the FC weight matrix.  





