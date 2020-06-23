# CS736-project
Course Project for the course CS 736

# Work to do

Start with a annotated 2D grayscale images dataset. 

- [ ] Shape modeling: Learning mean shape and modes of variations. (Save mean shape and eigenvectors)
- [ ] Texture modeling in mean shape space: Learning mean texture and modes of variatios (Save mean texture and eigenvectors). Use Delaunay Triangulation + Bilinear interpolation to transform texture to shape normalized frame and then solve for mean shape (by resolving lighting and illumination). 
- [ ] Combine texture and shape modeling. Find Ws and save paramters. Obtain combined modes of variations. 

- [ ] Fit model to a new image: Solve for image paramters, given the saved model paramters from earlier tasks. Obtain the best fit reconstruction and variations of reconstructions from analysis of different modes of variations. 
- [ ] Apply the algorithm for a 3D dataset or 2D color images dataset or both.

Things to consider: 
- Ws and R matrix computation is ad hoc in the original AAM paper. So that should be done differently.

