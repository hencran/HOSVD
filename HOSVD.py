#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTS
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import pandas as pd
from sklearn.decomposition import PCA
import torch 
import tensorly as tl
# settings for raster plots
from  matplotlib.colors import LinearSegmentedColormap

# SVD CLASS
'''
This class takes the SVD (np.linalg.svd) of an input matrix.
It also contains some built-in functions to visualize the SVD
'''
class SVD:
    def __init__(self, mat, s_as_array=False, full_matrices=True):     
        self.mat = mat
        self.s_as_array = s_as_array
        self.full_matrices = full_matrices
        
        if s_as_array:
            self.U, self.S, self.Vt, self.S_arr = self.svd(self.mat, self.s_as_array, self.full_matrices)
        else:
            self.U, self.S, self.Vt = self.svd(self.mat, self.s_as_array, self.full_matrices)
        
    def svd(self, mat, s_as_array=False, full_matrices=True):
        '''
        Performs the SVD on a matrix
        OUTPUTS: U, S, and Vt as matrices
        '''
        # Take SVD of input matrix
        U, S_arr, V = np.linalg.svd(mat, full_matrices)

        # Turn array into full S matrix
        S = np.zeros((U.shape[1], V.shape[0]))
        S[:S_arr.size, :S_arr.size] = np.diag(S_arr)

        if s_as_array:
            return U, S, V, S_arr
        else:
            return U, S, V
    def ReturnVals(self):
        if self.s_as_array:
            return self.U, self.S, self.Vt, self.S_arr
        else:
            return self.U, self.S, self.Vt

    def Visualize(self, params={}, num_vt_rows=3, aspect_mult=1, contrast=1):
        '''
        Visualizes the SVD by showing a 3x1 subplot
        with raster plot of U, bar chart of S, and line graph of rows in Vt
        '''
        # Unpack the parameters (if I want to plut a subset of the matrices, for example)
        if 'U' in params:
            U = params['U']
        else:
            U = self.U
        if 'S' in params:
            S = params['S']
        else:
            S = self.S
        if 'Vt' in params:
            Vt = params['Vt']
        else:
            Vt = self.Vt
        # no condition here for unexpected keys...
            
        # Prepare subplots
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.tight_layout()

        # U matrix (Raster plot)
        raster = self.RasterPlot(U, plot=False)
        ax[0].imshow(raster)
        ax[0].set_title("U")
        forceAspect(ax[0],aspect=aspect_mult*1.618033988749894)

        # S matrix (horizontal bar chart)
        pos = np.arange(np.shape(S)[1])
        ax[1].barh(pos, np.diagonal(S), align='edge')
        ax[1].set_ylim(len(pos), 0)  
        ax[1].set_title("S")

        # Vt matrix (line graph of select rows)
        color = iter(cm.rainbow(np.linspace(0, 1, num_vt_rows)))
        for i in range(num_vt_rows):
            c = next(color)
            ax[2].plot(self.Vt[i, :], color=c, marker='.', label=str(i))
        ax[2].set_title("Vt")
        plt.legend()
        plt.show()
        
    def RasterPlot(self, matrix, contrast=1, aspect_multiplier = 1, plot=True):
        '''
        Creates a raster plot with RGB colors
        *Note for all positive data, this will create poor plots
        '''
        # Get the dimensions of the matrix
        rows, columns = matrix.shape

        # Define contrast
        contrast = contrast

        # Create raster array
        raster = np.zeros((rows, columns, 3))
        for a in range(rows):
            for b in range(columns):
                rgbRaster = contrast * matrix[a, b]
                if rgbRaster > 0:
                    raster[a, b] = [rgbRaster, 0, 0]
                else:
                    raster[a, b] = [0, -rgbRaster, 0]

        # Create the plot
        if plot:
            fig, ax = plt.subplots()
            # g = plt.imshow(np.flip(raster, 0), aspect='equal')
            g = plt.imshow(raster, aspect='equal')
            #plt.gca().set_aspect('equal', adjustable='box')
            #plt.gca().set_data_ratio(1.61803398874989)
            self.forceAspect(ax, aspect_multiplier*1.61803398874989)
            plt.show()
        return raster
    
    def forceAspect(self, ax, aspect=1):
        '''
        Use to make raster plots look less squished
        '''
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


# In[ ]:


class HOSVD3(SVD):
    '''
    Class for calculating the HOSVD of a 3-D tensor. Follows the work in De Lathauwer
    Contains functions to visualize and extract different components of the HOSVD
    '''
    
    def __init__(self, Tensor):
        '''
        Expects a tensor of form (I1, I2, I3) or (X, Y, Z) 
        '''
        self.A = Tensor
        self.debug = False
        
        # dimensions of tensor
        self.I1, self.I2, self.I3 = self.TensorDimensions()
        
        # unfolded tensor
        self.A1 = None
        self.A2 = None
        self.A3 = None 

        # HOSVD unitary matrices
        self.U1 = None
        self.U2 = None
        self.U3 = None
        
        # HOSVD core
        self.S = None
        
        print('Tensor created, ready to decompose')
        print()
        
    def TensorDimensions(self):
        '''
        Extract the length of each dimension 
        '''
        s = self.A.shape
        return s[0], s[1], s[2]
    
    
    def Unfold(self):
        '''
        Unfolds the tensor into three matrices
        '''
        # A1
        self.A1 = tl.unfold(self.A, 0)
        if self.debug:
            print()
            print('Shape of A1: ', self.A1.shape)
            print('Expected Shape: ', (self.I1, self.I2*self.I3))
            
        # A2
        self.A2 = tl.unfold(self.A, 1)
        if self.debug:
            print()
            print('Shape of A2: ', self.A2.shape)
            print('Expected Shape: ', (self.I2, self.I1*self.I3))
        
        # A3      
        self.A3 = tl.unfold(self.A, 2)
        if self.debug:
            print()
            print('Shape of A3: ', self.A3.shape)
            print('Expected Shape: ', (self.I3, self.I1*self.I2))
            
    def GetU(self):
        '''
        Calculate the unitary matrices of the HOSVD (U^{(1)}, U^{(2)}, U^{(3)})
        '''
        # U1
        [self.U1, _, _] = super().svd(self.A1)
        
        if self.debug:
            print()
            print('Shape of U1: ', self.U1.shape)
            print('Expected Shape: ', (self.I1, self.I1))
            
        # U2
        [self.U2, _, _] = super().svd(self.A2)
        
        if self.debug:
            print()
            print('Shape of U2: ', self.U2.shape)
            print('Expected Shape: ', (self.I2, self.I2))
            
        # U3
        [self.U3, _, _] = super().svd(self.A3, full_matrices=False)
        
        if self.debug:
            print()
            print('Shape of U3: ', self.U3.shape)
            print('Expected Shape: ', (self.I3, self.I3))
        
    def GetS(self):
        '''
        Computes the core, S, of the tensor
        '''
        # compute the core unfolded in direction 1 (S_{(1)})
        S1 = np.dot(self.U1.T, np.dot(self.A1, np.kron(self.U2, self.U3)))
        L = np.shape(S1)[1]

        # Refold the core
        self.S = tl.fold(S1, 0, (self.I1, self.I2, self.I3))
        
        if self.debug:
            print()
            print('Shape of Core: ', self.S.shape)
            print('Expected shape: ', (self.I1, self.I2, self.I3))
    
    def PerformHOSVD(self):
        self.Unfold()
        self.GetU()
        self.GetS()

    def Check(self, unfold_dim='all'):
        '''
        Performs some tests to evaluate the algorithm's accuracy. 
        Can take some time to compute. Recommend using on 1 or 2 unfoldings
        '''
        # verify Unfolded matrices
        if unfold_dim == 'all':
            self.A1_check = np.dot(self.U1, np.dot(tl.unfold(self.S, 0), np.kron(self.U2, self.U3).T))
            self.A2_check = np.dot(self.U2, np.dot(tl.unfold(self.S, 1), np.kron(self.U1, self.U3).T))
            self.A3_check = np.dot(self.U3, np.dot(tl.unfold(self.S, 2), np.kron(self.U1, self.U2).T))
           
            print()
            print("Verifying work using De Lathauwer's equation 15:")
            print("\tA1 matches equation?", np.allclose(self.A1_check, self.A1))
            print("\tA2 matches equation?", np.allclose(self.A2_check, self.A2))
            print("\tA3 matches equation?", np.allclose(self.A3_check, self.A3))
            
        elif unfold_dim == 0:
            self.A1_check = np.dot(self.U1, np.dot(tl.unfold(self.S, 0), np.kron(self.U2, self.U3).T))
            print()
            print("Verifying work using De Lathauwer's equation 15:")
            print("\tA1 matches equation?", np.allclose(self.A1_check, self.A1))
            
        elif unfold_dim == 1:
            self.A2_check = np.dot(self.U2, np.dot(tl.unfold(self.S, 1), np.kron(self.U1, self.U3).T))
            print()
            print("Verifying work using De Lathauwer's equation 15:")
            print("\tA2 matches equation?", np.allclose(self.A2_check, self.A2))
            
        elif unfold_dim == 2:
            self.A3_check = np.dot(self.U3, np.dot(tl.unfold(self.S, 2), np.kron(self.U1, self.U2).T))
            print()
            print("Verifying work using De Lathauwer's equation 15:")
            print("\tA3 matches equation?", np.allclose(self.A3_check, self.A3))
            
    # Visualization Functions
    def ShowUnfoldings(self, aspect_mult=1, contrast=1):
        # Prepare subplots
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.tight_layout()

        # A1 (Raster plot)
        raster = super().RasterPlot(self.A1, plot=False, contrast=contrast)
        ax[0].imshow(raster)
        ax[0].set_title("A1")
        super().forceAspect(ax[0],aspect=aspect_mult*1.618033988749894)

        # A2 (Raster plot)
        raster = super().RasterPlot(self.A2, plot=False, contrast=contrast)
        ax[1].imshow(raster)
        ax[1].set_title("A2")
        super().forceAspect(ax[1],aspect=aspect_mult*1.618033988749894)
        
        # A3 (Raster plot)
        raster = super().RasterPlot(self.A3, plot=False, contrast=contrast)
        ax[2].imshow(raster)
        ax[2].set_title("A3")
        super().forceAspect(ax[2],aspect=aspect_mult*1.618033988749894)
        plt.show()
    
    def ShowU(self, aspect_mult=1, contrast=1):
        # Prepare subplots
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        plt.tight_layout()

        # A1 (Raster plot)
        raster = super().RasterPlot(self.U1, plot=False, contrast=contrast)
        ax[0].imshow(raster)
        ax[0].set_title("U1")
        super().forceAspect(ax[0],aspect=aspect_mult*1.618033988749894)

        # A2 (Raster plot)
        raster = super().RasterPlot(self.U2, plot=False, contrast=contrast)
        ax[1].imshow(raster)
        ax[1].set_title("U2")
        super().forceAspect(ax[1],aspect=aspect_mult*1.618033988749894)
        
        # A3 (Raster plot)
        raster = super().RasterPlot(self.U3, plot=False, contrast=contrast)
        ax[2].imshow(raster)
        ax[2].set_title("U3")
        super().forceAspect(ax[2],aspect=aspect_mult*1.618033988749894)
        plt.show()
    
    def ShowS(self, unfold_dim=0, aspect_mult=1, contrast=1):
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
        raster = super().RasterPlot(tl.unfold(self.S, unfold_dim), plot=False, contrast=contrast)
        plt.imshow(raster)
        super().forceAspect(ax,aspect=aspect_mult*1.618033988749894)
        #plt.colorbar()
        plt.title(f'$S_{unfold_dim}$')
        plt.show()

