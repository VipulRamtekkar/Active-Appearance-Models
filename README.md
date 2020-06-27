# CS736-project
Course Project for the course CS 736

# Work to do

Start with a annotated 2D grayscale images dataset. 

- [x] Shape modeling: Learning mean shape and modes of variations. (Save mean shape and eigenvectors)
- [x] Texture modeling in mean shape space: Learning mean texture and modes of variatios (Save mean texture and eigenvectors). Use Delaunay Triangulation + Bilinear interpolation to transform texture to shape normalized frame and then solve for mean shape (by resolving lighting and illumination). 
- [ ] Combine texture and shape modeling. Find Ws and save paramters. Obtain combined modes of variations. 

- [ ] Fit model to a new image: Solve for image paramters, given the saved model paramters from earlier tasks. Obtain the best fit reconstruction and variations of reconstructions from analysis of different modes of variations. [Done for Shapes]
