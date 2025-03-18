import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame  as pg
import numpy   as np
from scipy.signal import convolve2d
import time
import sys
import math


iKERNEL = 0
iC0 = 1
iH  = 2
iMU = 3
iSIGMA = 4


class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - dimensions est un tuple contenant le nombre d'éléments du maillage de l'espace dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est un array contenant des valeurs entre zéro et un initialisant la grille. 
    
    Si aucun pattern n'est donné, on tire au hasard pour chaque éléments du maillage un chiffre entre zéro et un
    Exemple :
       grid = Grille( (10,10), init_pattern=np.array([(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, R: int, channels: int, kernels: dict, init_pattern = None):
        import random
        
        if init_pattern is not None:
            self.dimensions = init_pattern.shape
            self.cells = init_pattern
            self.R = R
            self.kernels = kernels
            self.channels = channels
        
        else:
            dim = np.random.randint(200, 500, dtype=np.uint8)
            self.dimensions = (dim, dim)
            self.cells = np.random.uniform(0, 1, size=(dim, dim), dtype=np.double)
            self.R = 13
            self.channels = 1 

    @staticmethod
    def soft_clip(x, vmin, vmax):
      return 1 / (1 + np.exp(-4 * (x - 0.5)))


    @staticmethod
    def gauss(x, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError('s and sigma_filter must be positive values')
        return np.exp(-0.5 * ((x - mu) / sigma)**2)
  



    def K_lenia(self, mu_filter: float, sigma_filter: float, a = 'fft'):
        """
        If a is equal to 'fft' , it returns a list of lists of length equal to the number of channels.
        Each element contains the list of all the different growth functions applied to one channel.
        All the information on the growth function applied are of the form : [ filter, c0, h, m, s ]  
        
        If a is 'conv' it returns an array of size self.dimensions[: 2]
        """

        N, M = self.dimensions[: 2]

        if a == 'conv':
            R = self.R
            y, x = np.ogrid[-R : R , -R : R]
            distance = np.sqrt( (1 + x)**2 + (1 + y)**2 ) / R # 13 cells length equals to 1 unit of distance. 
            K_lenia = Grille.gauss(distance, mu_filter, sigma_filter)
            K_lenia[distance > 1] = 0
            K_lenia = K_lenia / np.sum(K_lenia)
            
        
        elif a =='fft' or a == 'target' or a == ['fft', 'fft', 'target'] or a == 'pacman':
            nb_channels = self.channels
            kernels = self.kernels
            K_lenia = [[]] * nb_channels
            y, x = np.ogrid[-N//2 : N//2, -M//2 : M//2]
            distance = np.sqrt( x**2 + y**2 ) / self.R 

            for kernel in kernels:
                nbr_rings = len( kernel["b"] )
                dist = distance * nbr_rings / kernel["r"] 
                filters = np.zeros((N,M))
                
                for i in range(nbr_rings):
                    ring =  kernel["b"][i] * Grille.gauss( dist - i , mu_filter, sigma_filter)
                    ring[ (dist >= (i + 1)) | (dist < i) ] = 0
                    filters = filters + ring

                filters = filters / np.sum( filters )
                filters = np.fft.fft2(np.fft.fftshift( filters ))
                K_lenia[kernel["c1"]] = K_lenia[kernel["c1"]] + [  [filters, kernel["c0"], kernel["h"], kernel["m"], kernel["s"]] ]
        

        else:
            print(f"Try 'convo', 'fft' instead of {a}")
            sys.exit()
       
        return K_lenia 




  
    
    def compute_next_iteration(self, K_lenia, a = 'fft', dt = 0.1):
        """

        """
        Vitalite = self.cells.copy()
        N, M = self.dimensions[: 2]
          

        if a == 'conv':
            mu_croissance = 0.15
            sigma_croissance = 0.015
            Energie = convolve2d( Vitalite, K_lenia, mode='same', boundary = 'wrap') # the first version which is too slow            
            G = -1 + 2*Grille.gauss(Energie, mu_croissance, sigma_croissance)


        
        elif a == 'fft' or a == 'target' or a == ['fft', 'fft', 'target'] or a == 'pacman':


            if self.channels == 1:
                G = np.zeros(self.dimensions)
                if a == 'fft':
                    for kernel in K_lenia[0]:
                        E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(self.cells))
                        E = np.real(E)
                        G +=  ( -1 + 2*Grille.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH] 
                else:
                    for kernel in K_lenia[0]:
                        E = np.fft.ifft2( kernel[iKERNEL]* np.fft.fft2(self.cells))
                        E = np.real(E)
                        G +=  ( Grille.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH]
                    G -= Vitalite


            elif self.channels > 1 :
                """ ATTENTION : ici filter est une liste à 3 dimensions -> 3 élements pour la première et 5 pour les autres"""
                
                G = np.zeros(self.dimensions, dtype = np.double)
                
                if type(a) == str:
                    for channel, channel_kernels in enumerate(K_lenia):
                        for kernel in channel_kernels:
                            E = np.fft.ifft2( kernel[iKERNEL] * np.fft.fft2(self.cells[:, :, kernel[iC0]] ))
                            E = np.real(E)
                            G[:,:,channel] += ( -1 + 2*Grille.gauss(E, kernel[iMU], kernel[iSIGMA]) ) * kernel[iH] 
                else:
                    for channel, channel_kernels in enumerate(K_lenia):
                        for kernel in channel_kernels:
                            E = np.fft.ifft2( kernel[iKERNEL] * np.fft.fft2(self.cells[:, :, kernel[iC0]] ))
                            E = np.real(E)
                            if channel != 2:
                                 G[:,:,channel] += ( -1 + 2*Grille.gauss(E, kernel[iMU], kernel[iSIGMA])) * kernel[iH]
                            else:
                                G[:, :, channel] +=  Grille.gauss(E, kernel[iMU], kernel[iSIGMA]) * kernel[iH]
                        G[:, :, 2] -= Vitalite[:, :, 2]




        else:
            print(f"Try 'convo', 'fft', 'pacman', 'target', instead of {a}")
            sys.exit()



        """
        Compute the next generation of cells :
        """
        if a != 'pacman':
            self.cells = np.clip( Vitalite + dt * G , 0,1)
        else:
            self.cells = Grille.soft_clip( Vitalite + dt * G , 0,1)

 


class Drawing:
    def __init__(self, width = 800, height = 600):
        self.colors = np.array([np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j]]).T
        self.dimensions = (width, height)
        self.screen = pg.display.set_mode(self.dimensions)#, pg.FULLSCREEN)

    def draw(self, cells, a):

        """
        Surface with pygame
        """
        if len(cells.shape) == 3: # there are more than one channel
            indices = np.array([(255*cells[:,:,i].T).astype(dtype=np.int32) for i in range(3)]).T

            # change colors for pacman pattern:
            if a == 'pacman':
                indices = np.roll(indices, -1, axis = 2)
                colors = np.array([self.colors[:,i][indices[:,:,i]] for i in range(3)])
            else:
                colors = np.array([self.colors[:,i][indices[:,:,i]] for i in range(3)])
            
            surface = pg.surfarray.make_surface(colors.T)
            surface = pg.transform.flip(surface, False, True)
            surface = pg.transform.scale(surface, self.dimensions)
            self.screen.blit(surface, (0,0))

        elif len(cells.shape) == 2:
            
            """ Trying to imitate matplotlib.pyplot.inferno """
            R = np.ogrid[0.:255.:256j]
            G = np.ogrid[0.:255.:256j]
            B = np.ogrid[0.:255.:256j]
            
            indices_B = ( 204 * np.sin(cells*np.pi*5/3)  ).astype(dtype=np.int32)
            indices_B[cells >= 0.60] = ( 130*(cells[cells >= 0.6]**6) ).astype(dtype=np.int32)

            indices_R = np.zeros_like(cells, dtype = np.int32)
            indices_R[cells<0.5] = (255*(1 - (1-(cells[cells<0.5]/0.5)**0.5 )**0.5)).astype(dtype=np.int32)
            indices_R[cells >= 0.5] = 255
            
            indices_G = (255 * (1- (1- (cells - 0.35)**2)**5)).astype(dtype=np.int32)
            indices_G[cells < 0.35] = 0
            colors = np.array([R[indices_R], G[indices_G], B[indices_B]]).T

            surface = pg.surfarray.make_surface(colors)
            surface = pg.transform.flip(surface, False, True)
            surface = pg.transform.scale(surface, self.dimensions)
            self.screen.blit(surface, (0,0))
        
        pg.display.update()
