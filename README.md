# Active Appearance Models

As part of course project for CS736 (IIT Bombay), we implemented Independent Active Appearance Model (in python) for IMM Frontal Face dataset, which has 120 annotated images of the frontal face (10 each of 12 different subjects). The images are annotated with 73 points.  We use 110 images for training and 10 images for testing. As the name suggests, our implementation models shape and texture independently. 

Active Appearance Model is a statistical model for linear modeling of appearance (texture) and shape (size, rotation, pose). They have been widely used for face modelling and in face recognition. 

## Shape Model 
We model shape on annotated pointsets. We find mean shape and covariance matrix using Procrustes Analysis for shape matching. The pointsets are transformed to preshape space, where we solve an alternate optimization problem to compute the mean shape and rotation matrix (to align shapes in preshape space to the mean shape). Out of (73x2 points) 146 eigenvectors, we retain 3 eigenvectors corresponding to the top 3 eigenvalues for shape fitting. Hence our shape model needs to compute 3 scalar values for fitting a test image. We use standard linear regression to iteratively fit the 3 coefficients for a test image. 

## Shape Normalization
Mean shape is taken as the frame of reference. For all images, the region within the pointset is transformed to the mesh of mean shape using Delaunay Triangulation, Bilinear Interpolation and Piecewise affine warping. From 800x600 image, we get a 232x240 shape normalized image patch.

## Texture Model
Texture is modeled in the mean shape mesh. The 232x240 image is further downsampled by a factor of 4 (to 58x60). To compute average texture, we again solve an alternate optimization problem for obtaining global light normalization parameters (defined for each image in terms of the mean texture) and mean texture. For texture fitting  model, we retain 60 eigenvectors out of (58x60) 3480 corresponding to the top 60 eigenvalues. So our texture model needs to compute 60 scalar values for fitting a test image. We again use linear regression (closed form solution of Maximum Likelihood estimate obtained via pseudoinverse). 

Therefore in total, our independent AAM is defined by a total of 63 parameters. 

# Execution Instructions 

To install the required libraries, run: 

`$ pip install -r requirements.txt`

To generate results, run:

`$ python main.py`

This will generate the following results:
- Shape Model:
  - Initial unaligned data
  - Mean shape with aligned data
  - Modes of variations (1,2,3)
  - Images closest to mean and first mode of variations
  - Reconstructions for test pointsets and RRMSE of the reconstructions
- Texture Model:
  - Mean Texture with aligned data
  - Modes of variations (1,2,3)
  - Reconstructions for test image textures and RRMSE of the reconstructions
- Combine modes of variations
  
The images in shape-normalized form (in the space of mean mesh) is already generated and provided with the repository. The code can be very easily modified to regenerate them as well.

# References:

## Papers
http://www2.imm.dtu.dk/pubdb/pubs/124-full.html     
https://link.springer.com/content/pdf/10.1023/B:VISI.0000029666.37597.d3.pdf    
https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/cootes-eccv-98.pdf    

## Code
https://github.com/krishnaw14/CS736-assignments     
https://github.com/TimSC/image-piecewise-affine

