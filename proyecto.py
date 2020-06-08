import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.feature import greycomatrix, greycoprops
import cv2  
from numpy.linalg import det
from skimage import util, exposure
from math import sqrt, ceil
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# global variables
black_value = np.float64(-1408.5106382978724) #valor para el color negro en formato de float 64
images_filename = 'tr_im.nii.gz' #conjunto de imagenes
masks_filename = 'tr_mask.nii.gz' #conjunto de mascáras

def get_vals(mask): #se recibe la imagen máscara 
    vals = [] #arreglo para almacenar los valores de la imagen de la mascara
    x, y = mask.shape #obtenemos las dimensiones de la  imagen de la mascara
    for i in range(x): 
        for j in range(y):
            if mask[i,j] not in vals: #si el valor no se encuentra en nuestro arreglo lo introducimos
                vals.append(mask[i][j])

    vals.remove(0)#removemos el valor 0 de nuestros valores
    return vals #regresamos el arreglo con los valores 1 2 3 4



def apply_mask(img, mask, class_id): #recibe la imagen, la máscara y la clase
    heigh, width = img.shape #obtenemos las dimensiones de la imagen
    tmp = deepcopy(img) #copiamos la imagen en una temporal
    lineal_array = np.array(0) # crea un arreglo con valor 0
    for x in range(heigh): 
        for y in range(width):
            # Establecer píxel como color negro si no está dentro de esta máscara
            if mask[x,y] != np.float64(class_id):
                tmp[x,y] = black_value
            # introducimos los valores que si estan dentro de la máscara
            else:
                lineal_array = np.append(lineal_array, tmp[x,y])
    # En tmp se guarda la imagen con la máscara aplicada
    # En lineal_array se guarda solo los píxeles de la máscara
    lineal_array = np.delete(lineal_array, 0) #se borran los valores de 0
    return tmp, lineal_to_matrix(lineal_array)


def calculate_media_lineal_arr(array):
    length = array.shape[0]
    media = np.float64(0)
    for x in range(length):
        media += array[x]
    media = media/length
    return media

def lineal_to_matrix(lineal_array):
    sqrt_ = ceil(sqrt(lineal_array.shape[0]))
    media = calculate_media_lineal_arr(lineal_array)
    missing_values = sqrt_**2 - lineal_array.shape[0]
    # Complete square matrix
    for x in range(missing_values):
        lineal_array = np.append(lineal_array, media)
    
    return lineal_array.reshape([sqrt_, sqrt_])

def show_slice_mask(slicei, mask): #mostrar la mascara y la imagen
    fig, ax = plt.subplots(1,2)    #el número de renglon y las columnas para mostrar la imagen
    ax[0].imshow(slicei.T,  cmap="gray", origin="lower") # mostrar la imagen rotada y en gris
    ax[0].set_title('Image')       #el titulo de la imagen
    ax[1].imshow(mask.T, cmap="gray", origin="lower")# mostrar la mascara rotada y en gris
    ax[1].set_title('Mask')        #el titulo de la imagen
    plt.show()

def glcm_properties(image): #recibe la imagen
    glcm = greycomatrix(image, distances = [1,2], angles = [0,-np.pi/2],symmetric=True, normed=True)
    Energia = greycoprops(glcm, 'energy')
    Homo = greycoprops(glcm, 'homogeneity')
    Contraste = greycoprops(glcm, 'contrast')
    print("E",Energia)
    print("H",Homo)
    print("C",Contraste)
    return [Energia,Homo,Contraste]

def main(): #función principal
    to_show = 3 
    imgs = nib.load(images_filename).get_fdata() #leemos el conjunto de imagenes
    masks = nib.load(masks_filename).get_fdata() #leemos el conjunto de mascáras
    energiaPro = np.zeros((2,2))
    homogeneidadPro = np.zeros ((2,2))
    contrastePro = np.zeros ((2,2))
    numEne = 0
    numHomo = 0
    numCon = 0
    for x in range(25, 25+to_show): #Ciclo para iterar las imagenes
        print("Image no ", x) #mensaje del número de imagen
        classes = get_vals(masks[:,:,x]) #Obtenemos los valores que tienen las mascáras
        show_slice_mask(imgs[:,:,x], masks[:,:,x]) #mostrar imagen y mascara
        for j in classes: #para mostrar el número de clases encontradas
            #tmp es la mascara aplicada y matrix sus pixeles
            tmp, matrix_mask = apply_mask(imgs[:,:,x], masks[:,:,x], j)
            print("class ", j)
            show_slice_mask(tmp, masks[:,:,x])
            #show_slice_mask(tmp, matrix_mask) #mostrar la mascara aplicada y sus pixeles
            Energia, Homogeneidad, Contraste = glcm_properties(matrix_mask.astype(np.uint8))
            if j == 1:
                numEne =numEne + 1
                for i in range(2):
                    for l in range(2):
                        energiaPro[i][l] = Energia[i][l]+energiaPro[i][l]              
            if j == 2:
                numHomo =numHomo+ 1
                for i in range(2):
                    for l in range(2):
                        homogeneidadPro[i][l] = Homogeneidad[i][l]+ homogeneidadPro[i][l]        
            if j == 3:
                numCon =numCon+ 1
                for i in range(2):
                    for l in range(2):
                        contrastePro[i][l] = Contraste[i][l]+ contrastePro[i][l]
                        
    for i in range(2):
        for l in range(2):
            energiaPro[i][l] = (energiaPro[i][l]/numEne)
            homogeneidadPro[i][l] = (homogeneidadPro[i][l]/numHomo)
            contrastePro[i][l] = (contrastePro[i][l]/numCon)
    print(numEne,energiaPro)
    print(numHomo,homogeneidadPro)
    print(numCon,contrastePro)

main()
