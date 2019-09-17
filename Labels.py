# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1459296,1401596,1461305
def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth

def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    scores = []
#    GT = np.array(GT)
#    scores = similarityMetric(description,GT[:,1], options)
    for i in range(0, len(description)):
        aux = similarityMetric(description[i],GT[i][1], options)
        scores.append(aux)
        
    mean_score = sum(scores) / len(description)
    return mean_score, scores



def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    
    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'
        
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    if options['metric'].lower() == 'basic'.lower():
        intersection = []
        intersection = set(Est).intersection(GT)
        S = float(len(intersection)) / float(len(Est))
        return S      
        
    else:
        return 0
        
def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """
       
#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
##  remind to create composed labels if the probability of 
##  the best color label is less than  options['single_thr']
    
    meaningful_colors = []
    unique = []
    j = 0
    
    for i in kmeans.centroids:
        if np.amax(i) < options['single_thr']:
            tmp = i.flatten()
            tmp.sort()
            main = cn.colors[np.where(i == tmp[-1])[0][0]]
            secondary = cn.colors[np.where(i == tmp[-2])[0][0]]
            
            if main < secondary:
                color = main + secondary
            else:
                color = secondary + main
                
            if color not in meaningful_colors:
                meaningful_colors.append(color)
                unique.append([j])
            else:
                unique[meaningful_colors.index(color)].append(j)
                
        else:
            if cn.colors[np.argmax(i)] in meaningful_colors:
                unique[meaningful_colors.index(cn.colors[np.argmax(i)])].append(j)
            else:
                meaningful_colors.append(cn.colors[np.argmax(i)])
                unique.append([j])
                
        j += 1
        
    return meaningful_colors, unique
            
            
#    maximos = kmeans.centroids.argmax(axis=1)
#    meaningful_colors = []
#    temp = []
#    unique = []
#    virgensita = []
#    madreSanta = [] 
#        
#    for i in maximos:
#        if i >= 0.6:
#            if i not in unique:
#                unique.append(i) 
#            temp.append(i)    
#        else:
#            senior.append(i)
#            
#    
#    unique.sort() 
#    
#    meaningful_colors = [cn.colors[i] for i in unique]
#    for i in range(0, len(unique)):
#        madreSanta = []
#        for x in range(0, len(temp)):
#            if temp[x] == unique[i]:
#                madreSanta.append(x)
#        virgensita.append(madreSanta)
#        
#    return meaningful_colors, virgensita
    
 #   kmeans.centroids -> Kx11


def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################
    #im = im.astype('uint8')
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():  
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'RGB'.lower():        
        pass #im = color.convert_colorspace(im, 'RGB', options['colorspace'])
    elif options['colorspace'].lower() == 'Lab'.lower():        
        im = im.astype('float64')
        im = color.rgb2lab(im/255)
    elif options['colorspace'].lower() == 'HSV'.lower():
        im = color.rgb2hsv(im.astype('uint8'))
        

##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the best K
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(im, options['K'], options) 
        kmeans.run()

     
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'RGB'.lower():        
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        
    elif options['colorspace'].lower() == 'Lab'.lower():
        kmeans.centroids = color.lab2rgb([kmeans.centroids])[0]*255
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
    elif options['colorspace'].lower() == 'HSV'.lower():
        kmeans.centroids = color.hsv2rgb([kmeans.centroids])[0]
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        
        
    
#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)   
    return colors, which, kmeans