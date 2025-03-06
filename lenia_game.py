import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame  as pg
import numpy   as np
from scipy.signal import convolve2d


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
    def __init__(self,R, canaux, kernels, init_pattern=None):# insérer le kernel ici? et R
        import random
        if init_pattern is not None:
            self.dimensions = init_pattern.shape
            self.cells = init_pattern
            self.energy = np.zeros(self.dimensions)
            self.R = R
            self.kernels = kernels
            self.canaux = canaux
        else:
            dim = np.random.randint(200, 500, dtype=np.uint8)
            self.dimensions = (dim, dim)
            self.cells = np.random.uniform(0,1, size=(dim, dim), dtype=np.double)
            self.R = 13


    @staticmethod
    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x-mu)/sigma)**2)
  


    def K_lenia(self, mu_filtre, sigma_filtre, a = 'fft'):  
        import sys

        N = self.dimensions[0]
        M = self.dimensions[1]

        if a == 'conv':
            """
            Filtre normal
            """
            R = self.R
            y, x = np.ogrid[-R:R , -R:R]
            distance = np.sqrt((1+x)**2 + (1+y)**2) / R # on considère que la distance séparant 13 éléments de la grille vaut 1. 
            K_lenia = Grille.gauss(distance, mu_filtre, sigma_filtre)
            K_lenia[distance > 1] = 0
            K_lenia = K_lenia / np.sum(K_lenia)
            
        elif a =='fft':
            R = self.R
            nb_canaux = self.canaux
            kernels = self.kernels
            K_lenia = [[]]*nb_canaux
            y, x = np.ogrid[-N//2:N//2, -M//2:M//2]
            distance = np.sqrt(x**2 + y**2) / R 

            for convo in kernels:
                dist = distance * len(convo["b"])
                filtre = np.zeros((N,M))
                
                for i in range(len(convo["b"])):
                    ring =  convo["b"][i] * Grille.gauss((dist - i*convo["r"])/convo["r"], mu_filtre, sigma_filtre)
                    ring[ (dist >= (i+1)*convo["r"]) | (dist < i * convo["r"]) ] = 0
                    filtre = filtre + ring
                filtre = filtre/np.sum(filtre)
                K_lenia[convo["c1"]] = K_lenia[convo["c1"]] + [ [filtre, convo["c0"], convo["h"], convo["m"], convo["s"]] ] # La liste de 3 elements contenants la liste des convolutions pour chaque canal.
                
        else:
            print("essayez 'convo', 'fft'à la place de {a}")
            sys.exit()
        
        return K_lenia 




  
    
    def compute_next_iteration(self, filtre, a = 'fft', dt = 0.1):
        import time
        import sys

       
        Vitalite = self.cells
        N = self.dimensions[0]
        M = self.dimensions[1]  
          

        """
        Calcul de l'Energie par convolution avec le filtre 
        """

        if a == 'conv':
            mu_croissance = 0.15
            sigma_croissance = 0.015
            Energie = convolve2d( Vitalite, filtre, mode='same', boundary = 'wrap') # the first version which is too slow            
            self.energy = Energie
            G = -1 + 2*Grille.gauss(Energie, mu_croissance, sigma_croissance)



        elif a == 'fft':
            if self.canaux == 1:
                G = []
                for kernel in filtre[0]:
                    E = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(kernel[0])) * np.fft.fft2(self.cells))
                    E = np.real(E)
                    G = G + [ ( -1 + 2*Grille.gauss(E, kernel[3], kernel[4]) ) * kernel[2] ]
                G =  np.mean(G,axis = 0)


            elif self.canaux == 3:
                """ ATTENTION : ici filtre est une liste à 3 dimensions -> 3 élements pour la première et 5 pour les autres"""
                V1 = self.cells[:, :, 0]
                V2 = self.cells[:, :, 1]
                V3 = self.cells[:, :, 2]
                
                V = [V1, V2, V3]
                
                G = [[]]*3
                for i, convos in enumerate(filtre):
                    for kernel in convos:
                        E = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(kernel[0])) * np.fft.fft2(V[kernel[1]] ))
                        E = np.real(E)
                        G[i] = G[i] + [ ( -1 + 2*Grille.gauss(E, kernel[3], kernel[4]) ) * kernel[2] ]
                    G[i] =  np.mean(G[i],axis = 0)
                G = np.array([G[0].T, G[1].T, G[2].T]).T
                
                test_filtre = np.zeros_like(Vitalite) 
                for k in range(3):
                    for i in range(5):
                        test_filtre[:, :, k] = test_filtre[:, :, k] + filtre[k][i][0]

                self.energy = test_filtre

        else:
            print("essayez 'convo', 'fft'à la place de {a}")
            sys.exit()



        """
        Calcule de la prochaine génération :
        """
        Vitalite = Vitalite + G*dt
        self.cells = np.clip(Vitalite, 0,1)
 





class Drawing:
    def __init__(self, width = 800, height = 600):
        self.colors = np.array([np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j]]).T
        self.dimensions = (width, height)
        self.screen = pg.display.set_mode(self.dimensions)#, pg.FULLSCREEN)

    def draw(self, cells):

        """
        Surface with pygame
        """
        if len(cells.shape) == 3:
            indices = np.array([(255*cells[:,:,i].T).astype(dtype=np.int32) for i in range(3)]).T
            colors = np.array([self.colors[:,i][indices[:,:,i]] for i in range(3)])
            surface = pg.surfarray.make_surface(colors.T)
            surface = pg.transform.flip(surface, False, True)
            surface = pg.transform.scale(surface, self.dimensions)
            self.screen.blit(surface, (0,0))

        elif len(cells.shape) == 2:
            
            """ J'essaye de refaire matplotlip.inferno à la main """
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
        else:
            print(f"problem in class Drawing")
            sys.exit()
        
        pg.display.update()


