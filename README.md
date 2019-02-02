# Reconnaissance de notes de musiques

## Objectifs
On souhaite convertir une image de partition musicale simple en fichier audio mp3. � cet effet, les �tapes suivantes ont �t� suivies : 
1. La constitution semi-automatis�e d'une base d'images labelis�es (voir dossier /Datasets) ;
2. L'augmentation de cette base de donn�es en appliquant des transformations aux images (flip de 180� p. ex.) ;
3. L'entra�nement d'un r�seau de neurones convolutif (CNN) pour d�tecter <u>si</u> (classification binaire) des notes sont pr�sentes sur les grid cells ;
4. Pour les images non �limin�es � l'�tape pr�c�dente, l'entra�nement d'une <i>random forest</i> pour pr�dire le type de note pr�sente sur l'image (classification multiclasse) ;
5. La pr�diction d'une nouvelle partition de test, en blendant de fa�on astucieuse les pr�dictions des deux mod�les ;
6. La production du fichier mp3 associ� � la partition test et sa lecture directement depuis le notebook.

Le d�tail des �tapes suivies est d�crit dans le power point Restitution.pptx.
Cette m�thode fonctionne bien pour des partitions simples et d'une qualit� suffisante. On peut ais�ment supposer qu'avec plus d'images, des partitions plus complexes pourraient �tre apprises.