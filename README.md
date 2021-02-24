# image scene classification project
<h2>Introduction</h2>
<p>La classification des scènes d'images de la nature, qui vise à étiqueter les images avec un
ensemble de catégories sémantiques basées sur leur contenu, a de larges applications dans
une gamme de domaines. Propulsée par les puissantes capacités d'apprentissage des
fonctionnalités des réseaux de neurones profonds, la classification des scènes d'images de la
nature, pilotée par l'apprentissage profond, a attiré une attention remarquable et a réalisé
des percées importantes. Cependant, un examen complet des récentes
réalisations en matière d'apprentissage en profondeur pour la classification des scènes
d'images de la nature fait encore défaut.
<br/>L’objectif du présent projet projet est de classifier automatiquement des images des scènes de la nature de manière efficace et efficiente, 
en 6 catégories telles que bâtiments, forêt, glacier, montagne, mer et rue en fonction des informations sémantiques des images
et en utilisant des méthodes d'apprentissage en profondeur (DL) notamment des modèles entièrement formés et d’autres pré-entraînés 
afin d’améliorer considérablement la précision de la classification.</p>
<h2>Problématique</h2>
<p>La classification des scènes est un travail important, qui peut fournir une base de prise de
décision pour le travail de suivi de la planification urbaine et de la surveillance
environnementale. La classification des scènes divise les images en diverses catégories telles
que les forêts, les villes et les rivières en fonction des informations sémantiques des images.
Les images sur lesquelles nous nous concentrons sont des images de scènes naturelles. Ces
images ont une résolution spatiale élevée, des structures géométriques et des motifs spatiaux
complexes. Cependant, il existe une forte similitude entre les catégories des images, tandis
que les images de la même catégorie de scène peuvent présenter de grandes différences.
Cela rend difficile la classification des scènes.  Il est à noter que pour certaines images, 
la modification d'un petit nombre de pixels peut affecter considérablement les résultats de classification, 
ce qui peut entraîner certains risques. Ceci est devenu un autre défi dans la classification des scènes. 
En raison de ces défis mentionnés ci-dessus, les modèles de classification devraient être supposés extraire très bien les informations sémantiques de l'image 
pour obtenir une meilleure robustesse.</p>
<h2>Descriptif du Dataset</h2>
Nom Dataset : Intel Image Classification. Image Scene Classification of Multiclass</br>
Contexte : il s'agit de données d'image de scènes naturelles du monde entier.</br>
Source du dataset : <a href="https://www.kaggle.com/puneet6060/intel-image-classification">https://www.kaggle.com/puneet6060/intel-image-classification</a></br>
Descriptif du dataset : Les données contiennent environ 25k images de taille 150x150 réparties en 6 catégories.</br>
{'bâtiments' -> 0, 'forêt' -> 1, 'glacier' -> 2, 'montagne' -> 3, 'mer' -> 4, 'rue' -> 5}</br>
Les données Train, Test et Prediction sont séparées dans chaque fichier zip. Il y a environ 14k images en train, 3k en test et 7k en prédiction.</br>

<h2>Modèles utilisés</h2>
<h3>CNN</h3>
<p>Les CNNs sont un type populaire des modèles du deep learning composé de couches
convolutives, de couches de regroupement, de couches entièrement connectées et d'une
couche softmax.</p>
<p>On a commencé par Création du modèle séquentiel 
c’est un moyen de créer des modèles d'apprentissage en profondeur dans
lesquels une instance de la classe Séquentiel est créée et des couches de modèle y sont créées
et ajoutées.
Puis on ajoute des Couche de convolution 2D : Cette couche crée un noyau de convolution qui est convolutionné avec l'entrée de la couche pour produire un tenseur de sorties. Lorsque nous utilisons cette couche comme première couche dans un modèle, on doit fournir l'argument input_shape. Parmi les paramètres obligatoires que
nous avons fourni à la classe Conv2D :</p>
• filters: Entier, la dimensionnalité de l'espace de sortie (c'est-à-dire le nombre de
filtres de sortie dans la convolution).</br>
Sur le premier ajout, nous apprenons un total de 32 filtres. Le pooling maximale est
ensuite utilisée pour réduire les dimensions spatiales du volume de sortie.
Nous apprenons ensuite 64 filtres sur la ligne 4. Là encore, le pooling max est utilisé
pour réduire les dimensions spatiales. La mise en pool maximale consiste à extraire la
valeur maximale de chaque mappe d'entités et à générer une couche de
regroupement. En raison de la simplicité de cette approche, elle a été largement
appliquée jusqu'à présent. La couche Conv2D finale apprend 128 filtres.
Nous remarquons que notre volume spatial de sortie diminue, notre nombre de filtres
appris augmente - c'est une pratique courante dans la conception d'architectures
CNN. En ce qui concerne le choix du nombre approprié de filtres, on a utilisé des
puissances de 2 comme valeurs.</br>
• kernel_size: un entier ou un tuple / une liste de 2 entiers, spécifiant la hauteur et la
largeur de la fenêtre de convolution 2D. Il peut s'agir d'un seul entier pour spécifier la
même valeur pour toutes les dimensions spatiales.</br>
• padding: l'un parmi "valide" ou "same" (insensible à la casse). «valide» signifie pas
de remplissage. «same» entraîne un remplissage uniforme vers la gauche / droite ou
vers le haut / le bas de l'entrée de sorte que la sortie ait la même dimension hauteur
/ largeur que l'entrée. Nous souhaitons plutôt conserver les dimensions spatiales du
volume de sorte que la taille du volume de sortie corresponde à la taille du volume
d'entrée, nous allons alors fournir une valeur "same" pour le padding.
</br>
• activation: Le paramètre d'activation de la classe Conv2D est simplement un
paramètre de commodité, vous permettant de fournir une chaîne spécifiant le nom
de la fonction d'activation que vous souhaitez appliquer après avoir effectué la
convolution.</br>
• kernel_regularizer: L'application de la régularisation nous
aide à: - Réduire les effets du overfitting, - Augmenter la capacité de notre modèle à
généraliser, - Lorsque nous travaillons avec de grands ensembles de données et des
réseaux de neurones profonds, l'application de la régularisation est généralement un
must.</br>
• strides : un entier ou un tuple / une liste de 2 entiers, spécifiant les strides de la
convolution le long de la hauteur et de la largeur. Il peut s'agir d'un seul entier pour
spécifier la même valeur pour toutes les dimensions spatiales. La valeur par défaut
des strides est (1, 1) cependant, on l’a augmenté à (2, 2) pour aider à réduire la taille
du volume de sortie.</br>
À la fin de la plupart des réseaux, nous ajoutons une couche entièrement connectée. Une
seule couche entièrement connectée avec 512 nœuds est ajoutée au CNN.
Enfin, un classificateur "softmax" est ajouté au réseau - la sortie de cette couche sont les
valeurs de prédiction elles-mêmes.</br>
Keras fournit un moyen de résumer un modèle. Le résumé est
textuel et comprend des informations sur : Les couches et leur ordre dans le modèle. La forme
de sortie de chaque couche, le nombre de paramètres (poids) dans chaque couche, le nombre
total de paramètres dans le modèle, le résumé peut être créé en appelant la fonction
summary () sur le modèle.</br>
Pour entraîner un modèle nous devons spécifier une fonction de perte, un optimiseur et,
éventuellement, des métriques à surveiller. Nous les transmettons au modèle en tant
qu'arguments de la méthode compile() :</br>
• loss function : La fonction de perte est utilisée pour trouver une erreur ou un écart
dans le processus d'apprentissage. </br>
• Optimizer : En apprentissage automatique, l'optimisation est un processus
important qui optimise les poids d'entrée en comparant la fonction de prédiction et
de perte. Keras fournit plusieurs optimiseurs sous forme de module. Dans notre cas,
on a appliqué l'optimiseur 'adam'.</br>
• metrics : les métriques sont utilisées pour évaluer les performances de votre
modèle. Il est similaire à la fonction de perte, mais n'est pas utilisé dans le processus
de formation.</br>
Nous appelons par la suite fit() , qui entraînera le modèle en découpant les données en "lots"
et en effectuant des itérations répétées sur l'ensemble de données pour un nombre donné.
Nous pouvons prédire la classe pour les nouvelles instances de données en utilisant notre
modèle de classification finalisé dans Keras en utilisant la fonction predict_classes ().</br>
À partir du graphique de la précision (accuracy), nous pouvons voir que le modèle pourrait
probablement être entraîné un peu plus car la tendance de la précision sur l'ensemble de
données est toujours en hausse pour les dernières époques (eposhs). Nous pouvons
également voir que le modèle n'a pas encore surappris l'ensemble de données
d'entraînement, montrant des compétences comparables sur les deux ensembles de
données.</br>
À partir du graphique de perte (loss), nous pouvons voir que le modèle a des performances
comparables sur les ensembles de données de train et de validation (test étiqueté). Si ces
parcelles parallèles commencent à s'écarter régulièrement, cela pourrait être un signe d'arrêt
de la formation à une époque antérieure.
<h3>VGG16 </h3>
<p>VGG16 est une architecture de réseau de neurones convolutifs qui a été finaliste du défi
ImageNet 2014 (ILSVR) avec une accuracy de test de 92,7% sur un ensemble de données de
14 millions d'images appartenant à 1000 classes. Bien qu'il ait terminé deuxième, il est devenu
un modèle de classification d'images assez populaire et est considéré comme l'une des
meilleures architectures de classification d'images.</p>
Implémenter un réseau de neurones avec Keras revient à créer un modèle Sequential et à
l'enrichir avec les couches correspondantes dans le bon ordre. L'étape la plus difficile est de
définir correctement les paramètres de chacune des couches – d'où l'importance de bien
comprendre l'architecture du réseau.</br>
Nous avons créé le modèle comme suit :</br>

Pour construire une couche de convolution, nous devons préciser le nombre de filtres utilisés,
leur taille, le pas et le zero-padding. Ils correspondent respectivement aux arguments filters ,
kernel_size , strides et padding du constructeur de la classe Conv2D. Son but est de repérer
la présence d'un ensemble de features dans les images reçues en entrée. Pour cela, on réalise
un filtrage par convolution : le principe est de faire "glisser" une fenêtre représentant la
feature sur l'image, et de calculer le produit de convolution entre la feature et chaque portion
de l'image balayée.</br>
Seulement deux valeurs sont possibles pour padding : 'same' ou 'valid'. La couche de
convolution réduit la taille du volume en entrée avec l'option 'valid', mais la conserve avec
'same'. Pour VGG-16, on utilisera donc toujours l'option 'same'.</br>
S'il s'agit de la toute première couche de convolution, il faut préciser dans l'argument
input_shape les dimensions des images en entrée du réseau. Pour notre cas, input_shape =
(150,150,3).</br>
Enfin, pour indiquer la présence d'une couche ReLU juste après la couche de convolution, on
ajoute l'argument activation = 'relu'. La couche de correction ReLU remplace donc toutes les
valeurs négatives reçues en entrées par des zéros. Elle joue le rôle de fonction d'activation.</br>
On ajoute des couches de batch normalisation qui effectuent un recentrage et une
normalisation des données. De cette façon, il n'y a pas beaucoup de changement dans chaque
entrée de couche. Ainsi, les couches du réseau peuvent apprendre simultanément, sans
attendre que la couche précédente apprenne. Cela accélère l’entrainement des réseaux. Le momentum est fixé à une valeur élevée d'environ 0,9. Un momentum si élevé entraînera un
apprentissage lent mais régulier.</br>
Une couche de pooling est définie par la taille des cellules de pooling et le pas avec lequel on
les déplace. Ces paramètres sont précisés dans les arguments respectifs pool_size et strides
du constructeur de la classe MaxPooling2D. Cette couche permet de réduire le nombre de
paramètres et de calculs dans le réseau. On améliore ainsi l'efficacité du réseau et on évite le
sur-apprentissage.</br>
A ce stade, nous pouvons déjà implémenter quasiment tout le réseau VGG-16. Il ne reste plus
qu'à ajouter les couches fully-connected, en créant des objets de la classe Dense. Ce type de
couche reçoit en entrée un vecteur 1D. Il faut donc convertir les matrices 3D renvoyées par
la dernière couche de pooling. Pour cela, on instancie la classe Flatten juste avant la première
couche fully-connected.</br>
L'argument units du constructeur de Dense permet de préciser la taille du vecteur en sortie.
De plus, une correction ReLU ou softmax est effectuée juste après la couche fully-connected,
on l'indique dans le paramètre activation.
Ainsi, les trois dernières couches fully-connected et leur fonction d'activation (ReLU pour les
deux premières, softmax pour la dernière) sont ajoutées.

<h2>InceptionV3</h3>
<p>InceptionV3 est la 3ème version d'une série d'architectures convolutionnelles Deep Learning. Inception V3 a été entrainé à l'aide d'un ensemble de données de 1 000 classes à partir de l'ensemble de données ImageNet original qui a été formé avec plus d'un million d'images d'entraînement. InceptionV3 a été entrainé pour le challenge ImageNet Large Visual Recognition, où il était un premier finaliste.</p>
Nous allons d’abord utiliser toutes les couches du modèle à l'exception de la dernière couche
entièrement connectée car elle est spécifique à ImageNet. Après nous allons rendre toutes
les couches non entrainables.
La suppression des dernières couches se fait en ajoutant l'argument include_top = False lors
de l'import du modèle pré-entraîné. Dans ce cas, il faut aussi préciser les dimensions des
images en entrée (input_shape ) :</br>
On procède ensuite à :</br>
• Aplatir la couche de sortie à 1 dimension</br>
• Ajoutez une couche entièrement connectée avec 1024 unités cachées et activation ReLU</br>
• Ajouter un taux d'abandon de 0,6</br>
• Ajouter une dernière couche softmax pour la classification</br>
Nous devons maintenant "compiler" notre modèle avec la méthode model_v3.compile(), puis
l'entraîner avec model_v3.fit().

<h3>MobileNetV2</h3>
<p>Les MobileNets sont basés sur une architecture simplifiée qui utilise des convolutions séparables en profondeur pour créer des réseaux neuronaux profonds légers. L'idée de base derrière Mobile Net v1 était de remplacer les convolutions coûteuses par des convolutions moins chères. C'était un grand succès. Le principal changement dans l'architecture v2 a été l'utilisation de blocs de goulots d'étranglement inversés et de connexions résiduelles.</p>
On commence par la création du modèle de base à partir du modèle pré-entraîné MobileNetV2.</br>
La fonction de création du modèle a des paramètres par défaut que nous avons modifié à notre convenance :</br>
• input_shape : taille image d’entrée</br>

• include_top: Boolean, s'il faut inclure la couche entièrement connectée en haut du
réseau. La valeur par défaut est True.</br>
• weights: le chemin vers le fichier de poids à charger.</br>

Maintenant que nous avons notre configuration de couche de base, nous pouvons ajouter le
classificateur. Au lieu d'aplatir la carte de caractéristiques de la couche de base, nous utiliserons une
couche de regroupement moyenne globale qui fera la moyenne de la zone 5x5 entière de
chaque carte d'entités 2D et nous renverra un seul vecteur de 1280 éléments par filtre.</br>

<h2>Résultats </h2>
Dans ce travail, nous évaluons et analysons deux stratégies possibles d'exploitation des modèles: </br>
(i)	des modèles entièrement formés, comme le CNN que nous avons créé à partir de zéro et le VGG-16 que nous avons implémenté.</br>
(ii)	des modèles pré-entrainé utilisés comme extracteurs de caractéristiques. Comme l’InceptionV3 et le MobileNet V2</br>
Dans la première stratégie (i), un réseau est formé à partir de zéro pour obtenir des caractéristiques visuelles spécifiques pour l'ensemble de données. Cette approche est préférable car elle donne un contrôle total de l'architecture et des paramètres, ce qui tend à produire un réseau plus robuste et efficace. Cependant, cela nécessite une quantité considérable de données, car la convergence du réseau est élaguée au sur-ajustement (Overfitting). Cet inconvénient rend presque impossible la conception et la formation complètes d'un réseau à partir de zéro pour la plupart des problèmes de classification de scènes.
Les réseaux de neurones convolutifs (CNN) appliqués à la classification des scènes présentent deux problèmes communs. La première est que ces modèles ont un grand nombre de paramètres, ce qui entraîne facilement un sur-ajustement (Overfitting). L'autre est que le réseau n'est pas assez profond, donc des informations sémantiques plus abstraites ne peuvent pas être extraites. 
En plus de cela les deux modèles de la stratégie (i) et même le modèle InceptionV3 de la stratégie (ii) n’ont pas abouti à une bonne performance pour pouvoir faire une prédiction pertinente et précise.</br>
Pour résoudre ces problèmes, nous avons proposé un réseau complet simple et efficace basé sur MobileNet pour la classification des scènes qui rend le réseau plus profond, mais n'augmente pas significativement le nombre de paramètres. Ce modèle a pu aboutir à une meilleure performance par rapport aux autres modèles.</br>
Pour résoudre le problème de Overfitting, on a essayé plusieurs méthodes comme la régularisation, l'ajout de couches Dropout, le Transfer Learning. On a même essayé plusieurs architectures du réseau CNN mais quand on arrive à surmonter le problème du Overfitting un autre problème apparaît; l'accuracy du test est plus grande que celle du train et le loss du train est plus grand que celui du test.</br>
En fait, Il existe une forte similitude entre les catégories des images par exemple dans les images de la catégorie "Street" on trouve des "Buildings" ce qui rend difficile de détecter s'il s'agit d'une image de catégorie Street ou Building même pour l'Humain. Ce problème persiste même pour les catégories "Glacier" et "Mountain". Tandis que les images de la même catégorie de scène peuvent présenter de grandes différences. Cela rend difficile une classification des scènes pertinente.
<h2>Conclusion </h2>
Dans le cadre de ce projet, la performance des modèles pré-formés (VGG16, Inception V3 et MobileNet V2) et d'un CNN entièrement formés hautement reconnus sur la tâche de classification est évaluée pour l'ensemble de notre Dataset d'images de scènes naturelles. Les relations entre la taille de l'ensemble de données d'entraînement, le nombre d'époques pour la formation, le nombre de couches CNN et les paramètres apprenables sont étudiées en profondeur.</br>
Pour le réseau CNN, nous avons pu couvrir les concepts suivants : 	</br>
•	ConvNet se compose d'un extracteur de caractéristiques et d'un réseau neuronal de classification. Son architecture de couche profonde avait été un obstacle qui a rendu le processus de formation difficile. Cependant, depuis que le Deep Learning a été introduit comme solution à ce problème, l'utilisation de ConvNet s'est rapidement développée.</br>
•	L'extracteur de caractéristiques de ConvNet consiste en des piles alternées de la couche de convolution et de la couche de regroupement. Comme ConvNet traite des images bidimensionnelles, la plupart de ses opérations sont menées dans un plan conceptuel bidimensionnel.</br>
•	À l'aide des filtres de convolution, la couche de convolution génère des images qui accentuent les caractéristiques de l'image d'entrée. Le nombre d'images de sortie de cette couche est le même que le nombre de filtres de convolution que contient le réseau. Le filtre de convolution n'est en fait qu'une matrice bidimensionnelle.</br>
•	La couche de regroupement (pooling) réduit la taille de l'image. Il lie les pixels voisins et les remplace par une valeur représentative. La valeur représentative est la valeur maximale ou moyenne des pixels.</br>
•	Nous avons montré que, malgré les performances impressionnantes des CNN sur des images distribuées de manière similaire aux données d'entraînement, leur précision est bien inférieure sur les images négatives. Nos observations indiquent que les CNN qui sont simplement entraînés sur des données brutes réussissent mal à reconnaître la sémantique des objets.</br>
En l'absence de variance entre les images d'entraînement, l'augmentation du nombre d'échantillons d'images contribue non seulement au temps de calcul sans améliorer les performances, mais augmente également le risque de surajustement (Overfitting) à mesure que le nombre d'images avec des caractéristiques similaires analysées par époque augmente. Pour résoudre le problème de du surajustement, nous avons essayé une panoplie de méthodes comme la régularisation, l'ajout de couches Dropout et le Transfer Learning. </br>
En outre, on observe que les réseaux avec des connexions de couche hiérarchique et des couches multiples entièrement connectées fonctionnent mieux dans des conditions variables et semblent prometteurs au cours de la réalisation d'un cadre de classification.</br>
En conclusion, les réseaux pré-entraînés ont une applicabilité élevée sur la détection et la classification des images même s'ils sont formés sur des ensembles de données complètement différents en raison des fonctionnalités de bas niveau. On observe que les caractéristiques apprises au cours de la formation sont transférables avec une grande précision. En outre, le nombre requis d’échantillons d’apprentissage en moins et les réseaux de convergence rapide font des réseaux pré-entraînés une option favorable pour la mise en œuvre de CNN pour la tâche de classification des images de scènes.
