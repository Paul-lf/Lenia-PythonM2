"""
Projet inspiré du script python de Science étonnante
Lenia
###############
Lenia est un automate cellulaire qui peut être vu comme une généralisation du jeu de la vie inventé par Conway 
Pour rappel le jeu de la vie se base normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonales.) L'interaction se fait selon les
règles suivantes pour calculer l'irération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Lenia généralise en attribuant une valeur entre zéro et un pour chaque élément de la grille. On donne à cette valeur le nom de vitalité
car on peut la voir comme le pourcentage de l'espace qu'occupe une cellule vivante. Les règles du jeu sont alors différentes:
    - On utlise un filtre pour attribuer à chaque cellule une "Energie" (ici on fait une convolution avec un filtre en anneau)
    - On ajoute alors à la vitalité une valeur en fonction de l'énergie de la cellule à l'aide d'une fonction de croissance
    - La nouvelle valeur est raaporté à l'intervalle [0, 1] avec la focntion np.clip()

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
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
    def __init__(self, init_pattern=None):
        import random
        if init_pattern is not None:
            self.dimensions = init_pattern.shape
            self.cells = init_pattern
        else:
            dim = np.random.randint(200,dtype=np.uint8)
            self.dimensions = (dim, dim)
            self.cells = np.random.uniform(2, size=dim, dtype=np.uint8)


    @staticmethod
    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x-mu)/sigma)**2)


    def compute_next_iteration(self):

        dt= 0.1
        mu_filtre = 0.5
        sigma_filtre = 0.15
        mu_croissance = 0.15
        sigma_croissance = 0.015
        
        """
        Calcule la prochaine génération :
        """
        Vitalite = np.zeros(self.dimensions, dtype=np.uint8)
        Vitalite = self.cells
        
        """
        Création du filtre en anneau
        """
        R = 13 
        y, x = np.ogrid[-R:R, -R:R]
        distance = np.sqrt((1+x)**2 + (1+y)**2) / R # on considère que la distance séparant 13 éléments de la grille vaut 1.
        K_lenia = Grille.gauss(distance, mu_filtre, sigma_filtre)
        K_lenia[distance > 1] = 0               # Cut at d=1
        K_lenia = K_lenia / np.sum(K_lenia)     # Normalize
        
        """
        Equivalent au calcul de voisins pour le jeu de la vie
        """
        Energie = convolve2d(self.cells, K_lenia, mode='same', boundary = 'wrap')
        # print(f"ENergie: {[np.max(Energie),np.min(Energie), np.median(Energie), np.mean(Energie)]}")

        
        """
        Fonction de croissance
        """
        G = -1 + 2*Grille.gauss(Energie, mu_croissance, sigma_croissance)
        
        self.cells = self.cells + G*dt
        self.cells = np.clip(self.cells, 0,1)


class Drawing:
    def __init__(self, width = 800, height = 600):
        self.colors = np.array([np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j], np.ogrid[0.:255.:256j]]).T
        self.dimensions = (width, height)
        self.screen = pg.display.set_mode(self.dimensions)

    def draw(self, cells):
        indices = (255*cells).astype(dtype=np.int32)
        surface = pg.surfarray.make_surface(self.colors[indices.T])
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, self.dimensions)
        self.screen.blit(surface, (0,0))
        pg.display.update()


if __name__ == '__main__':
    
    import time
    import sys
    
    pg.init()

    """
    orbium
    """
    N = 256
    M = int(np.ceil((16*N)/9))
    orbium =  np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
    grille_orbium = np.zeros((N,M))
    pos_x = M//6
    pos_y = N//6
    grille_orbium[pos_x:(pos_x + orbium.shape[1]), pos_y:(pos_y + orbium.shape[0])] = orbium.T   
    orbium_init = grille_orbium

    """
    Gaussian spot centerd in the middle
    """
    N = 512
    M = int(np.ceil((16*N)/9))
    radius = 36
    y, x = np.ogrid[-N//2:N//2, -M//2:M//2]
    grille_gauss = np.exp(-0.5 * (x*x + y*y) / (radius*radius))
    gaussian_spot = grille_gauss


    dico_patterns = {'orbium': orbium_init, 'gaussian_spot': gaussian_spot }

    choice = 'gaussian_spot'
    if len(sys.argv) > 1 :
        choice = sys.argv[1]
    print(f"Pattern initial choisi : {choice}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)
    grid = Grille(init_pattern)
    
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
        appli = Drawing( width = resx, height = resy)
        print(f"resolution ecran : {resx,resy}")
    else:
        appli = Drawing()

    
    mustContinue = True
    while mustContinue:
        
        t1 = time.time()
        diff = grid.compute_next_iteration()
        t2 = time.time()
        #time.sleep(500) # A régler ou commenter pour vitesse maxi
        appli.draw(grid.cells)
        t3 = time.time()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                mustContinue = False
        print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='');
    pg.quit()
