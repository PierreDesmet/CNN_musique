# Reconnaissance de notes de musiques

## Objectifs
On souhaite convertir une image de partition musicale simple en fichier audio mp3. À cet effet, les étapes suivantes ont été suivies : 
1. La constitution semi-automatisée d'une base d'images labelisées (voir dossier /Datasets) ;
2. L'augmentation de cette base de données en appliquant des transformations aux images (flip de 180¡ p. ex.) ;
3. L'entraînement d'un réseau de neurones convolutif (CNN) pour détecter <u>si</u> (classification binaire) des notes sont présentes sur les grid cells ;
4. Pour les images non éliminées à l'étape précédente, l'entraînement d'une <i>random forest</i> pour prédire le type de note présente sur l'image (classification multiclasse) ;
5. La prédiction d'une nouvelle partition de test, en blendant de façon astucieuse les prédictions des deux modèles ;
6. La production du fichier mp3 associé à la partition test, et sa lecture directement depuis le notebook.

Le détail des étapes suivies est décrit dans le power point Restitution.pptx.
Cette méthode fonctionne bien pour des partitions simples et d'une qualité suffisante. On peut aisément supposer qu'avec plus d'images, des partitions plus complexes pourraient être apprises.   
