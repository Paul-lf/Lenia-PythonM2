"""
Lenia is a mathematical life form that is a generalization of Conway's game of life to a continuous point of view. The values that pixels can take is called
vitality because it can be thought of as the density of living pixels inside it. We assign a value between zero and one for each element of the grid.

The rules of the game are changed to make time and space seem continuous: 
    - We use ring shaped filters to assign to each pixels an "Energy" with a convolution. This generalizes the rule where the state of neighbours influences
the state of a pixel at the next iteration.
    - The energy of the pixel is transformed by either a growth or target function in our model which acts like a gradient of time for the vitality. 
    - The new value for vitality is restricted to the interval [0, 1] with the function np.clip() 

For this project, we change the rules slightly by transforming the infinite grid into a torus containing a finite number of pixels. The leftmost pixels have the rightmostpixels as neighbors and vice versa, and in the same way the pixels that are highest have the pixels that are lowest as neighbors and vice versa.

Bibliography:
- https://colab.research.google.com/github/OpenLenia/Lenia-Tutorial/blob/main/Tutorial_From_Conway_to_Lenia.ipynb
- https://github.com/JuvignyEnsta/ProjetPythonM2_2025
- https://github.com/scienceetonnante/lenia/
"""

# NOTE: soft_clip -> smooth the visualization of data ; an explicit euler scheme 

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame  as pg
import numpy   as np
from scipy.signal import convolve2d
import sys
import math

iKERNEL = 0
iC0 = 1
iH  = 2
iMU = 3
iSIGMA = 4



class Grid:
    """
    Input parameters to create a Grid object:
        - init_pattern is an array containing values between 0 to 1.
        - Channels is the number of channels (One channel for each coordinate if we use a third dimension for the Grid object).
        Each channel can interact with each other through the growth or target functions)
        - kernels is a list of dictionnaries containing all the informations needed for the convolution (cf examples_lenia.py)
        - R is the number of pixels that represent one unit of length
    
    If no pattern is given a number between 0 to 1 is randomly given to each pixels.
    Example :
       grid = Grid( R=13, channels=1, kernels=[{"b":[1], "m":0.15, "s":0.015, "h":1, "r":1, "c0":0, "c1":0}], init_pattern=np.array([(2,2),(0,2),(4,2),(2,0),(2,4)] )
    """
    def __init__(self, R: int, channels: int, kernels: list, init_pattern = None):
        import random
        if (channels<=0 or R<=0):
            raise ValueError("channels and R are integers greater than 0")
        if init_pattern is not None:
            self.dimensions = init_pattern.shape
            self.pixels = init_pattern
            self.pixels_pts_milieux = init_pattern
            self.R = R
            self.kernels = kernels
            self.channels = channels
            self.nbr_iter = 0
        
        else:
            dim = np.random.randint(200, 500, dtype=np.uint8)
            self.dimensions = (dim, dim)
            self.pixels = np.random.uniform(0, 1, size=(dim, dim), dtype=np.double)
            self.R = 13
            self.channels = 1 

    @staticmethod
    def soft_clip(x): # Alternate method to deal with values above 1 or below 0 when computing the next iteration of pixels that gives smoother output.
      return 1 / (1 + np.exp(-4.15 * (x - 0.5)))


    @staticmethod
    def gauss(x, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError('s and sigma_filter must be positive values')
        return np.exp(-0.5 * ((x - mu) / sigma)**2)
  



    def K_lenia(self, mu_filter: float, sigma_filter: float, a = 'fft'):
        """
        If a is equal to 'fft' , it returns a list of lists of length equal to the number of channels.
        Each element contains the list of all the different growth or target functions applied to one channel.
        All the informations on the growth function applied to a channel are listed in the following order: [ filter, c0, h, m, s ]  
        
        If a is 'conv' it returns an ndarray of size self.dimensions[: 2]
        """

        N, M = self.dimensions[: 2]

        if a == 'conv':
            y, x = np.ogrid[-self.R : self.R , -self.R : self.R]
            distance = np.sqrt( (1 + x)**2 + (1 + y)**2 ) / self.R # 13 pixels length equals to 1 unit of length. 
            K_lenia = Grid.gauss(distance, mu_filter, sigma_filter)
            K_lenia[distance > 1] = 0
            K_lenia = K_lenia / np.sum(K_lenia)
            
        
        elif a =='fft' or a == 'target' or a == ['fft', 'fft', 'target'] or a == 'pacman':
            K_lenia = [[]] * self.channels
            y, x = np.ogrid[-N//2 : N//2, -M//2 : M//2]
            distance = np.sqrt( x**2 + y**2 ) / self.R 

            for kernel in self.kernels:
                nbr_rings = len( kernel["b"] )
                dist = distance * nbr_rings / kernel["r"] 
                filters = np.zeros((N,M))
                
                for i in range(nbr_rings):
                    ring =  kernel["b"][i] * Grid.gauss( dist - i , mu_filter, sigma_filter)
                    ring[ (dist >= (i + 1)) | (dist < i) ] = 0
                    filters = filters + ring

                filters = filters / np.sum( filters )
                filters = np.fft.fft2(np.fft.fftshift( filters ))
                K_lenia[kernel["c1"]] = K_lenia[kernel["c1"]] + [  [filters, kernel["c0"], kernel["h"], kernel["m"], kernel["s"]] ]
        

        else:
            print(f"Try 'convo', 'fft', 'target' or   ['fft', 'fft', 'target'] instead of {a}")
            sys.exit()
       
        return K_lenia 




    
    def compute_next_iteration(self, K_lenia, a = 'fft', dt = 0.1):
        """
        Computes self.pixels
        """
        Vitalite = self.pixels.copy()
        N, M = self.dimensions[: 2]
          

        if a == 'conv':
            mu_croissance = 0.15
            sigma_croissance = 0.015
            if self.nbr_iter%2==0:
                Energie = convolve2d( Vitalite, K_lenia, mode='same', boundary = 'wrap') # the first version which is too slow            
                G = (-1 + 2*Grid.gauss(Energie, mu_croissance, sigma_croissance))/2
            else:
                Energie = convolve2d( self.pixels_pts_milieux, K_lenia, mode='same', boundary = 'wrap')
                G = -1 + 2*Grid.gauss(Energie, mu_croissance, sigma_croissance)


        
        elif a == 'fft' or a == 'target' or a == ['fft', 'fft', 'target'] or a == 'pacman':

            if self.channels == 1:
                G = np.zeros(self.dimensions)
                if a == 'fft':
                    for kernel in K_lenia[0]:
                        E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(Vitalite))
                        E = np.real(E)
                        G +=  ( -1 + 2*Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH]  
                    ## Uncomment to try "point milieux instead" of Euler scheme. Comment four lines above.
                    #if self.nbr_iter%2==0:
                     #   for kernel in K_lenia[0]:
                      #      E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(Vitalite))
                       #     E = np.real(E)
                        #    G +=  ( -1 + 2*Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH] / 2 
                    #else:
                     #   for kernel in K_lenia[0]:
                      #      E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(self.pixels_pts_milieux))
                       #     E = np.real(E)
                        #    G +=  ( -1 + 2*Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH] 
                else:
                    for kernel in K_lenia[0]:
                        E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(Vitalite))
                        E = np.real(E)
                        G +=  ( Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH]
                    G -= Vitalite


            elif self.channels > 1 :
                """ WARNING : filter is a three dimension list -> 3 elements for the first and 5 for the two others"""
                
                G = np.zeros(self.dimensions, dtype = np.double)
                
                if type(a) == str: # If we use a growth function in all channels
                    for channel, channel_kernels in enumerate(K_lenia):
                        for kernel in channel_kernels:
                            E = np.fft.ifft2( kernel[iKERNEL] * np.fft.fft2(self.pixels[:, :, kernel[iC0]] ))
                            E = np.real(E)
                            G[:,:,channel] += ( -1 + 2*Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH] 
                else:
                    for channel, channel_kernels in enumerate(K_lenia):
                        for kernel in channel_kernels:
                            E = np.fft.ifft2( kernel[iKERNEL] * np.fft.fft2(self.pixels[:, :, kernel[iC0]] ))
                            E = np.real(E)
                            if channel != 2: 
                                 G[:,:,channel] += ( -1 + 2*Grid.gauss(E, kernel[iMU], kernel[iSIGMA])) * kernel[iH]
                            else: # We only use this option with 'pacman' where the blue channel uses a target function instead of a growth function
                                G[:, :, channel] +=  Grid.gauss(E, kernel[iMU], kernel[iSIGMA]) * kernel[iH]
                        G[:, :, 2] -= Vitalite[:, :, 2]



        """
        Compute the next generation of pixels :
        """
        if a != 'pacman' and a!='conv': # and a!='fft': # Try "point milieux" scheme with orbium for instance(uncomment lines 162-171)
            self.pixels = np.clip(Vitalite + dt * G , 0,1)
        
        elif a == 'pacman':
            self.pixels = Grid.soft_clip(Vitalite + dt * G)
        
        else: # Here we use the "point milieux" scheme and not the explicit Euler scheme 
            if self.nbr_iter%2 == 0:
                self.pixels_pts_milieux = np.clip(Vitalite + dt * G, 0, 1)
            else:
                self.pixels = np.clip(Vitalite + dt * G, 0, 1)
        self.nbr_iter += 1


class Drawing:
    def __init__(self, width = 800, height = 600):
        self.colors = np.array([np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j]]).T
        self.dimensions = (width, height)
        self.screen = pg.display.set_mode(self.dimensions)

    def draw(self, pixels, a):
        """
        Surface with pygame
        """
        if len(pixels.shape) == 3: # More than one channel
            indices = np.array([(255*pixels[:,:,i]).astype(dtype=np.int32) for i in range(3)])

            # change colors for pacman pattern:
            if a == 'pacman':
                indices = np.roll(indices, -1, axis = 0)
                colors = self.colors[:,0][indices]
            else:
                colors = self.colors[:,0][indices]
            
            surface = pg.surfarray.make_surface(colors.T)
            surface = pg.transform.flip(surface, False, True)
            surface = pg.transform.scale(surface, self.dimensions)
            self.screen.blit(surface, (0,0))


        elif len(pixels.shape) == 2:
            """ Trying to imitate matplotlib.pyplot.inferno() """
            R = np.ogrid[0.:255.:256j]
            G = np.ogrid[0.:255.:256j]
            B = np.ogrid[0.:255.:256j]
            
            indices_B = ( 204 * np.sin(pixels*np.pi*5/3)  ).astype(dtype=np.int32)
            indices_B[pixels >= 0.60] = ( 130*(pixels[pixels >= 0.6]**6) ).astype(dtype=np.int32)

            indices_R = np.zeros_like(pixels, dtype = np.int32)
            indices_R[pixels<0.5] = (255*(1 - (1-(pixels[pixels<0.5]/0.5)**0.5 )**0.5)).astype(dtype=np.int32)
            indices_R[pixels >= 0.5] = 255
            
            indices_G = (255 * (1- (1- (pixels - 0.35)**2)**5)).astype(dtype=np.int32)
            indices_G[pixels < 0.35] = 0
            colors = np.array([R[indices_R], G[indices_G], B[indices_B]]).T

            surface = pg.surfarray.make_surface(colors)
            surface = pg.transform.flip(surface, False, True)
            surface = pg.transform.scale(surface, self.dimensions)
            self.screen.blit(surface, (0,0))
        
        pg.display.update()
