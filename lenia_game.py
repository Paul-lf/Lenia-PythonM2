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
            dim = np.random.randint(200,500,dtype=np.uint8)
            self.dimensions = (dim, dim)
            self.cells = np.random.uniform(0,1, size=(dim, dim), dtype=np.double)
            self.R = 13


    @staticmethod
    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x-mu)/sigma)**2)
  


    def K_lenia(self, mu_filtre, sigma_filtre, a = 'conv'):  
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

        elif a == 'fft':
            """
            Filtre pour la fft
            """
            R = self.R
            yfft, xfft = np.ogrid[-N//2:N//2, -M//2:M//2]
            distancefft = np.sqrt(xfft**2 + yfft**2) / R # on considère que la distance séparant 13 éléments de la grille vaut 1.
            K_leniafft = Grille.gauss(distancefft, mu_filtre, sigma_filtre)
            K_leniafft[distancefft > 1] = 0
            K_leniafft = K_leniafft / np.sum(K_leniafft)
            K_lenia = K_leniafft
        
        elif a == 'multi rings':
            """
            filtre à plusieurs anneaux (pour l'hydrogenium)
            """
            R = self.R
            yfft, xfft = np.ogrid[-N//2:N//2, -M//2:M//2]
            distancefft = np.sqrt(xfft**2 + yfft**2) / R*3 # on considère que la distance séparant 13 éléments de la grille vaut 1.

            ring_1 = 0.5 * Grille.gauss(distancefft, mu_filtre, sigma_filtre)
            ring_1[distancefft >= 1] = 0


            ring_2 = Grille.gauss(distancefft-1, mu_filtre, sigma_filtre) # distancefft -1 permet de calculer pour r-1 dans [0; 1[ quand r dansv[1; 2[
            ring_2[(distancefft < 1)|(distancefft >= 2)] = 0


            ring_3 = 0.667 * Grille.gauss(distancefft -2, mu_filtre, sigma_filtre)
            ring_3[(distancefft < 2)|(distancefft >= 3)] = 0 

            K_lenia = ring_1 + ring_2 + ring_3

            K_lenia = K_lenia/np.sum(K_lenia)
        
        elif a == 'multi growth':
            """
            fish -> mouvements plus erratiques (comme des demi-tours)
            """
            """ Convolution 1 """
            R = self.R

           # kernels = [
            #        {"b":[1, 5/12, 2/3], "m":mu_filtre, "s":sigma_filtre, "h":1/3, "r":1, "c0":0, "c1":0 },
            #        {"b":[1/12, 1], "m":mu_filtre, "s":sigma_filtre, "h":1/2, "r":1, "c0":0, "c1":0 },
            #        {"b":[1], "m":mu_filtre, "s":sigma_filtre, "h":1, "r":1, "c0":0, "c1":0 }]
            
            y, x = np.ogrid[-N//2:N//2, -M//2:M//2]
            distance = np.sqrt(x**2 + y**2) / R 

            ring_1 = Grille.gauss(distance*3, mu_filtre, sigma_filtre)
            ring_1[distance*3 >= 1] = 0


            ring_2 = 5*Grille.gauss(distance*3-1, mu_filtre, sigma_filtre)/12 
            ring_2[(distance*3 < 1)|(distance*3 >= 2)] = 0


            ring_3 = 2 * Grille.gauss(distance*3 -2, mu_filtre, sigma_filtre)/3
            ring_3[(distance*3 < 2)|(distance*3 >= 3)] = 0 

            K_lenia1 = ring_1 + ring_2 + ring_3
            K_lenia1 = K_lenia1/np.sum(K_lenia1)
            
            """ Convolution 2 """

            ring_1 = Grille.gauss(distance*2, mu_filtre, sigma_filtre)/12
            ring_1[distance*2 >= 1] = 0

            ring_2 = Grille.gauss(distance*2-1, mu_filtre, sigma_filtre) 
            ring_2[(distance*2 < 1)|(distance*2 >= 2)] = 0

            K_lenia2 = ring_1 + ring_2 
            K_lenia2 = K_lenia2/np.sum(K_lenia2)

            """ Convultion 3 """

            K_lenia3 = Grille.gauss(distance, mu_filtre, sigma_filtre)
            K_lenia3[distance >= 1] = 0
            K_lenia3 = K_lenia3/np.sum(K_lenia3)
            K_lenia = [K_lenia1, K_lenia2, K_lenia3]

        elif a =='multi canaux':
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
            sys.exit()
        
        return K_lenia 




  
    
    def compute_next_iteration(self, filtre, a = 'conv', dt = 0.1):
        import time
        import sys

        mu_croissance = 0.15
        sigma_croissance = 0.015


        Vitalite = self.cells
        
        """
        ETAPE 1:
        Création du filtre en anneau
        """

        
        N = self.dimensions[0]
        M = self.dimensions[1]  
          

        """
        ETAPE 2:
        Calcul de l'Energie par convolution avec le filtre 
        """
        if a == 'conv':
            Energie = convolve2d( Vitalite, filtre, mode='same', boundary = 'wrap') # the first version which is too slow            
            self.energy = Energie
            G = -1 + 2*Grille.gauss(Energie, mu_croissance, sigma_croissance)

        elif a == 'fft' or a == 'multi rings':

            Energiefft = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(filtre)) * np.fft.fft2(Vitalite ))
            Energiefft = np.real(Energiefft)
            self.energy = Energiefft
            if a == 'fft':
                G = -1 + 2*Grille.gauss(Energiefft, mu_croissance, sigma_croissance)
            else:
                G = -1 + 2*Grille.gauss(Energiefft,0.26, 0.036)

        elif a == 'multi growth':

            Energie1 = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(filtre[0])) * np.fft.fft2(Vitalite ))
            Energie1 = np.real(Energie1)
            #self.energy = Energie1
            G1 = -1 + 2*Grille.gauss(Energie1, 0.156, 0.0118)
            
            Energie2 = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(filtre[1])) * np.fft.fft2(Vitalite ))
            Energie2 = np.real(Energie2)
            #self.energy = Energie2
            G2 = -1 + 2*Grille.gauss(Energie2, 0.193, 0.049)
            
            Energie3 = np.fft.ifft2( np.fft.fft2(np.fft.fftshift(filtre[2])) * np.fft.fft2(Vitalite ))
            Energie3 = np.real(Energie3)
            #self.energy = Energie3
            G3 = -1 + 2*Grille.gauss(Energie3, 0.342, 0.0891)
            
            G = np.mean([G1, G2, G3], axis=0)
        
        elif a == 'multi canaux':
            
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
            print("essayez 'convo', 'multi', 'fft' ou 'canaux' à la place de {a}")
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


