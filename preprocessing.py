import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image as pil_image

def tri_images(path='Binaire/Test/*.png'):
    """
    Retourne la liste des chemins d'images du dossier @path dans l'ordre croissant de numéro
    """
    import re, glob
    li = glob.glob(path)
    chiffres_idx = [int(re.findall(r'[0-9]+', x)[0]) for x in li]
    dico = dict(zip(li, chiffres_idx))
    return sorted(li, key=lambda x : dico[x])



def get_label_from_files(files_list, mode='binary'):
    """
    binary      : classification binaire, ex. "il y-t-il une note de musique sur cette image ?"
    multiclasse : classification multiclasse, ex. "quelle est la note de musique présente sur cette image"
    """
    if mode=='binary':
        return np.array(['0.png' not in f for f in files_list]).astype(int)
    if mode=='multiclasse':
        import re
        return np.array([re.search(r"[\d]+_(?P<NOTE>[a-zA-Z_0]+)\.png", f).group('NOTE') for f in files_list])
        #return np.array([f.split('_')[1].replace('.png','') for f in files_list])

def plot_roc_curve(y_true, y_pred):
    from sklearn.metrics import roc_curve, roc_auc_score
    plt.figure(figsize=(4,3))
    AUC = roc_auc_score(y_true=y_true, y_score=y_pred)
    fpr, tpr, threshold = roc_curve(y_true=y_true, y_score=y_pred)
    plt.plot(fpr, tpr, color='lightblue', lw=2, label='ROC curve (AUC = %0.3f)' % AUC)
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve\n')
    plt.legend(loc="lower right")



def preprocesse_image(path):
    img = mpimg.imread(path)
    if len(img.shape) > 2:
        if img.shape[2] > 3:
            img = np.delete(img, 1, np.s_[-1])
    if len(img.shape) > 2:
        img_noir_blanc = list(map(lambda x : [z>0.8 for y in x for z in y], img))
        img_noir_blanc = np.array(img_noir_blanc).reshape((img.shape[0], img.shape[1], 3))
        img_noir_blanc = np.array([[y[0] for y in x] for x in img_noir_blanc]).reshape((img.shape[0], img.shape[1]))
    else :
        img_noir_blanc = img


    for seuil, épaisseur_portée in [(0.1, 1), (0.45, 2), (0.66, 3)]:
        abscisses = []
        for i in np.arange(0, img_noir_blanc.shape[0], épaisseur_portée):
            if np.sum(img_noir_blanc[i,:]) < seuil*img_noir_blanc.shape[1]:
                abscisses.append(i)
        if len(abscisses) %5 == 0 : break
    
    def get_taille_note(abscisses):
        diff = 0
        for i in range(len(abscisses)-1):
            diff += abscisses[i+1]-abscisses[i]
        largeur_note = int(diff / (len(abscisses)-1))
        hauteur_note = (max(abscisses) - min(abscisses))*(3/4) # une note = 3/4 de portée
        return largeur_note, hauteur_note

    largeur_note, hauteur_note = get_taille_note(abscisses)
    print('Une note de la partition prend en moyenne', largeur_note, 'pixels de large sur', hauteur_note, 'pixels de haut.')


    pas = largeur_note//3
    centres = []

    a = min(abscisses)-largeur_note
    b = max(abscisses)+largeur_note

    nb_pixels = pas*pas*4
    for j in np.arange(pas, img_noir_blanc.shape[1], pas):
        for i in np.arange(a, b, pas):
            nb_pixels_éteints = np.sum(img_noir_blanc[i-pas:i+pas, j-pas:j+pas])
            nb_pixels_allumés = nb_pixels - nb_pixels_éteints
            if nb_pixels_allumés > nb_pixels-(largeur_note/4):
                centres.append((i,j))
    print(len(centres), "points noirs ont été détectées.")
    
    
    
    def drop_doublons(centres):
        centres_finaux = []
        for i in range(len(centres)-1):
            if np.abs(centres[i][1] - centres[i+1][1]) > largeur_note:
                centres_finaux.append(centres[i])
        centres_finaux.append(centres[-1])
        return centres_finaux

    centres = drop_doublons(centres)
    print(len(centres), "points noirs uniques ont été détectées.")
    
    
    
    ordonnées = []
    for j in range(img_noir_blanc.shape[1]):
        nb_pixel_considérés = len(img_noir_blanc[a:b,j])
        sum_pixels = nb_pixel_considérés - np.sum(img_noir_blanc[a:b,j])
        # Si le nombre de pixels est suffisamment grand pour qu'il s'agisse d'une note...
        if sum_pixels > (3/6)*nb_pixel_considérés :
            ordonnées.append(j)# ... et suffisamment petit pour que ce ne soit pas un séparateur...
            
            
    # On interdit que deux barres verticales détectées se suivent à moins de largeur_note pixels
    ordonnées_kept = []
    for i in range(0, len(ordonnées)):
        if ordonnées[i] - ordonnées[i-1] > largeur_note:
            ordonnées_kept.append(ordonnées[i-1])
    ordonnées_kept.append(ordonnées[-1]) # ajout de la dernière note
    
    
    centres_finaux = []
    for centre in centres:
        for ordonnée in ordonnées_kept:
            if np.abs(centre[1] - ordonnée) < largeur_note:
                centres_finaux.append(centre)
    print(len(centres_finaux),'notes ont été détectées.')
                
                
    # Enregistrement des images
    for i in range(len(centres_finaux)):
        plt.figure(figsize=(6,4))
        note = centres_finaux[i]
        c = note[1]-largeur_note
        d = note[1]+largeur_note
        plt.imsave('A_predire/'+str(i)+'.png', img_noir_blanc[a:b, c:d])
        plt.axis('off')
        plt.close()
        
    # Et redimensionnement final
    import re 
    for i in [f for f in os.listdir('A_predire/') if f.endswith('.png') and re.match(r'[0-9]+', f)]:
        im = pil_image.open('A_predire/'+i)
        im = im.resize((14,56))
        im.save('A_predire/'+i)