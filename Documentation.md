# SVD and HOSVD3 Class Documentation
##SVD Class
Purpose
The SVD class performs Singular Value Decomposition (SVD) on a given matrix and provides methods to visualize the decomposition.

### Initialization
``` 
def __init__(self, mat, s_as_array=False, full_matrices=True)
```
mat: The input matrix to decompose.
s_as_array: If True, returns the singular values as an array.
full_matrices: If True, computes the full-sized U and Vt matrices.
### Methods
####svd

```
def svd(self, mat, s_as_array=False, full_matrices=True)
```
Performs SVD on the input matrix.

mat: The input matrix.
s_as_array: If True, returns the singular values as an array.
full_matrices: If True, computes the full-sized U and Vt matrices.
Returns U, S, and Vt matrices (and S_arr if s_as_array is True).

####ReturnVals
```
def ReturnVals(self)
```
Returns the decomposed matrices U, S, Vt, and optionally S_arr.

####Visualize
```
def Visualize(self, params={}, num_vt_rows=3, aspect_mult=1, contrast=1)
```
Visualizes the SVD by showing a 3x1 subplot with raster plot of U, bar chart of S, and line graph of rows in Vt.

params: Dictionary to specify subsets of matrices for plotting.
num_vt_rows: Number of rows from Vt to plot.
aspect_mult: Aspect multiplier for the plots.
contrast: Contrast for the raster plot.

####RasterPlot
```
def RasterPlot(self, matrix, contrast=1, aspect_multiplier=1, plot=True)
```
Creates a raster plot with RGB colors.

matrix: The input matrix.
contrast: Contrast for the raster plot.
aspect_multiplier: Aspect multiplier for the plot.
plot: If True, displays the plot.

####forceAspect
```
def forceAspect(self, ax, aspect=1)
```
Adjusts the aspect ratio of raster plots to make them look less squished.

ax: Axis object to adjust.
aspect: Desired aspect ratio.

##HOSVD3 Class
Purpose
The HOSVD3 class performs Higher-Order Singular Value Decomposition (HOSVD) on a 3-D tensor and provides methods to visualize the decomposition.

###Initialization
```
def __init__(self, Tensor)
```
Tensor: The input 3-D tensor for decomposition.

###Methods
####TensorDimensions
```
def TensorDimensions(self)
```
Extracts the length of each dimension of the tensor.

####Unfold
```
def Unfold(self)
```
Unfolds the tensor into three matrices corresponding to each mode.

####GetU
```
def GetU(self)
```
Calculates the unitary matrices of the HOSVD.

####GetS
```
def GetS(self)
```
Computes the core tensor S of the HOSVD.

####PerformHOSVD
```
def PerformHOSVD(self)
```
Performs the full HOSVD process, including unfolding the tensor and calculating U and S.

####Check
```
def Check(self, unfold_dim='all')
```
Performs tests to evaluate the accuracy of the HOSVD algorithm.

unfold_dim: Dimension(s) to check. Can be 0, 1, 2, or 'all'.

###Visualization Functions

####ShowUnfoldings
```
def ShowUnfoldings(self, aspect_mult=1, contrast=1)
```
Shows raster plots of the unfolded matrices.

####ShowU
```
def ShowU(self, aspect_mult=1, contrast=1)
```
Shows raster plots of the unitary matrices.

####ShowS
```
def ShowS(self, unfold_dim=0, aspect_mult=1, contrast=1)
```
Shows a raster plot of the core tensor unfolded in the specified dimension.
