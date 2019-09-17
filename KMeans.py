
"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
#import math

    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    dist = np.zeros((X.shape[0],C.shape[0]))
#    suma=0
#    for i in range(X.shape[0]):
#        for j in range(C.shape[0]):
#            dif = X[i]-C[j]
#            for e in dif:
#                suma = suma + e**2
#            dist[i,j] = math.sqrt(suma)
#            suma = 0
#    return dist

    for i in range(C.shape[0]):
        dif = X-C[i]
        dif = dif**2
        dif = dif.sum(axis=1)
        dist[:,i] = dif**0.5
        
    return dist
    
    #return np.random.rand(X.shape[0],C.shape[0])

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """
#        self._init_X(X)
        self._init_X(X.astype('float64'))                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################
        self.clusters = np.zeros(len(self.X))
        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if (len(X.shape)>2):
            self.X= X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        else:
            self.X= X.reshape(X.shape[0], X.shape[1])
        

            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if self.K>0:
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        self.num_iter = 0                                      # INT current iteration
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        pixeles= []
        j = 0
        #print "Seleccionar K =",self.K
        if self.options['km_init'] == 'first':
            while (len(pixeles) != self.K):
                aux = self.X[j]
                if j == 0:
                    pixeles.append(aux)
                else:
                    insert = False
                    for element in pixeles:
                        if np.array_equal(element, aux):
                            insert = True
                    if not insert:
                        pixeles.append(aux)
                j+=1
            self.centroids = np.array(pixeles);
        else:
            self.centroids = np.random.rand(self.K,self.X.shape[1])
        
#        print "CENTROIDES: ", self.centroids
        ##self.centroids = np.random.rand(self.K,self.X.shape[1])
        
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        distancia = distance(self.X, self.centroids)
        
#        for i in range (distancia.shape[0]):
#            minimaDistancia = min(distancia[i])
#            self.clusters[i] = distancia[i].tolist().index(minimaDistancia)
            
        self.clusters = distancia.argmin(axis=1)
#            print self.clusters

#        print "CLUSTERS:", self.clusters
        #self.clusters = np.random.randint(self.K,size=self.clusters.shape)

        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        """self.old_centroids = np.copy(self.centroids)
        suma = np.zeros((self.K, self.X.shape[1]))
        sumK = np.zeros(self.K)
#        aux = self.clusters[0]
        for i in range (self.X.shape[0]):
            suma[int(self.clusters[i])] += self.X[i]
            sumK[int(self.clusters[i])]+= 1
        
        for y in range(self.K):
            if sumK[y]==0:
                self.centroids[y] = 0
            else:
                self.centroids[y] = suma[y]/sumK[y]
        """
        self.old_centroids =  np.copy(self.centroids) 
        for i in range(self.K): 
            x=self.X[self.clusters==i, :] 
#            if x.size > 0: 
            self.centroids[i,:] = np.mean(x, 0) 

        return        

    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if (self.options['tolerance'] == 0):
            np.isnan(self.centroids)
            self.centroids[np.isnan(self.centroids)]=0
            return np.array_equal(self.centroids, self.old_centroids)
        else:
            """min_old_centroids = np.copy(self.old_centroids)
            min_old_centroids -= self.options['tolerance']
            max_old_centroids = np.copy(self.old_centroids)
            max_old_centroids += self.options['tolerance']
            if ( (self.centroids >= min_old_centroids) and (self.centroids <= max_old_centroids) ):
                return True
            else:
                return False
            """
            #aux = np.zeros(self.centroids.shape[1]);
            
            
            dif = self.centroids - self.old_centroids
            dif = dif**2
            dif = dif.sum(axis=1)
            dif = dif**0.5
            
#            dif = dif.mean()
            canvi = dif < self.options['tolerance']
            
            return canvi.all()
                        
#            aux = self.centroids[(self.centroids>(self.old_centroids - self.options['tolerance']))*(self.centroids<(self.old_centroids + self.options['tolerance']))]
#            return np.all(aux)
        
        #return np.array_equal(self.centroids, self.old_centroids)
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K==0:
            self.bestK()
            return        
        
        self._iterate(True)
        self.options['max_iter'] = np.inf
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        
        fit = []
        
        for i in range(2,5):
            self._init_rest(i)
            self.run()
            fit.append(self.fitting())
            
        return fit.index(min(fit)) + 2
#        self._init_rest(4)
#        self.run()        
#        fit = self.fitting()
#        return 4
#        nuevaK = self.K+1
#        
##        self.run()
#        aux = []
#        for i in range(nuevaK):
#            self._init_rest(i)
#            fit = self.fitting()
#            aux.append(fit)
#        B = aux.index(max(aux))+1
#        
#        return B

        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        if self.options['fitting'].lower() == 'fisher':
            distancias = np.zeros((self.K,1))
            fila = 0
            contador = np.zeros((self.centroids.shape[0],1))
            dFisher = np.zeros((self.centroids.shape[0],1))
            intraClass = 0
            puntMitja=np.zeros((1,self.X.shape[1]))
            interClass = 0
            bestK=0
            
            for i in self.clusters:
                distancias[i] = distancias[i]+np.linalg.norm(self.X[fila] - self.centroids[i])
                contador[i] = contador[i]+1
                fila = fila+1
            
            for i in range(dFisher.shape[0]):
                dFisher[i] = distancias[i]/contador[i]
                intraClass=intraClass + dFisher[i]
                
            intraClass = intraClass/dFisher.shape[0]
            
            puntMitja = self.X.sum(axis = 0)
            
            for i in range(self.centroids.shape[0]):
                interClass = interClass + np.linalg.norm(self.centroids[i] - puntMitja)
                
            interClass = interClass/self.K
            bestK = intraClass/interClass
            
            return bestK
        else:
            return np.random.rand(1)
        
#        betweenVariance = 1/self.K
#        
#        for k in range(self.K):
#            betweenVariance *= 1/self.clusters[k]
#            for i in range(self.clusters):
               

    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'	
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
#            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)