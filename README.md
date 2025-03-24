
# Projet : Maintenance Prédictive avec Réseau de Neurones  
  
## 1. Dataset utilisé  
  
Le projet s’appuie sur le jeu de données **AI4I 2020 Predictive Maintenance Dataset**, contenant 10 000 observations. Chaque ligne représente un état de fonctionnement d’une machine industrielle à un instant donné, avec :  
  
- Des variables mesurées par des capteurs tel que la température, l'usure de l'outil, la vitesse de rotation, le couple...  
- Un label indiquant si une panne est survenue  
- Les types de pannes correspondants si une panne s'est produite.  
  
Dès lors 5 types de panne sont possibles et sont décrites ci-dessous :  
  
- `TWF` : Tool Wear Failure — Panne due à l’usure de l’outil  
- `HDF` : Heat Dissipation Failure — Panne de dissipation thermique  
- `PWF` : Power Failure — Panne électrique  
- `OSF` : Overstrain Failure — Panne due à une surcharge  
- `RNF` : Random Failure — Panne aléatoire  
  
## 2. Visualisation du dataset  
  
Nous avons commencé par charger le jeu de données afin de comprendre sa structure, de vérifier les colonnes , les diiférents types de données, la nature des valeurs...  
  
```python  
import pandas as pd  
  
df = pd.read_csv("ai4i2020.csv")  
print(df.head())  
```  
En effet, La fonction `read_csv()` permet de lire un fichier CSV et de le convertir en **DataFrame**, une structure optimisée pour l’analyse de données.  
  
L’instruction `print(df.head())` affiche les cinq premières lignes du jeu de données.  
  
## 3. Analyse de la distribution des pannes  
  
Nous avons analysé la répartition de la variable appelée `Machine failure`, qui indique si une machine est tombée en panne ou non .  
  
  
  
```python  
  
failure_counts = df['Machine failure'].value_counts()  
  
# On visualise la distribution  
plt.figure(figsize=(6, 4))  
plt.bar(failure_counts.index, failure_counts.values,  
color=['green', 'red'],  
tick_label=['No Failure', 'Failure'])  
  
plt.xlabel('Machine Failure')  
plt.ylabel('Count')  
plt.title('Distribution of Machine Failures and Non-Failures')  
plt.show()  
```  
  
  
- `df['Machine failure'].value_counts()` : permet de compter combien de fois chaque valeur (`0` ou `1`) apparaît dans la colonne `Machine Failure`  
  
- `plt.bar()` : nous utilisons un diagramme en barres pour visualiser et comparer les machines qui fonctionnent et celles qui sont en panne.  
  
  
L’analyse de la variable `Machine failure` montre un déséquilibre marqué entre les machines en bon état et celles ayant rencontré une panne. La grande majorité des données concerne des machines n'étant pas en panne ce qui peut biaiser l’apprentissage du modèle.  
  
En effet, ce déséquilibre peut conduire le modèle à ignorer les cas des pannes qui sont minoritaires , en apprenant à prédire uniquement l’absence de panne.  
  
  
#### Répartition globale des types de panne  
  
Nous avons ensuite cherché à étudier la fréquence de chaque type de panne (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) dans l’ensemble du dataset pour repérer les types de pannes les plus fréquentes dans l'environnement industriel.  
Nous avons donc réalisé un **diagramme à barres** à l’aide du code python suivant, qui cherche à visualiser la fréquence de chaque type de panne .  
  
```python  
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] # Types de pannes à analyser  
failure_counts = df[failure_types].sum() # Somme des valeurs pour chaque panne  
  
plt.figure(figsize=(8, 5))  
bars = plt.bar(failure_counts.index, failure_counts.values,  
color=['red','blue','green','yellow','purple']) # Une couleur par type  
  
for bar in bars:  
height = bar.get_height()  
plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',  
ha='center', va='bottom', fontsize=12, fontweight='bold') # Valeur exacte affichée  
  
plt.xlabel('Failure Type')  
plt.ylabel('Count')  
plt.title('Distribution of Different Failure Types')  
plt.xticks(rotation=45)  
plt.show()  
```  
  
Dès lors, nous remarquons que certaines pannes (comme `HDF` ou `OSF`) sont bien plus fréquentes que d’autres tel que `RNF` qui se font très rares, ce qui aura un impact sur un éventuel apprentissage multi-classe.  
  
---  
  
#### Répartition des pannes uniquement parmi les machines défaillantes  
  
Pour affiner l’analyse, nous avons modifié le code du diagramme de pannes précédent afin de nous concentrer uniquement sur les machines ayant réellement rencontré une panne donc quand `Machine failure == 1`.  
  
Nous sommes donc partis du code précédent , qui affichait la distribution globale des pannes et avons modifié les lignes suivantes :  
```python  
failed_machines = df[df['Machine failure'] == 1]  
```  
Cette ligne permet de filtrer **les machines défaillantes**, dans le but d'étudier la répartition réelle des types de panne parmi les machines qui ont connu une panne.  
``` python  
failure_counts = failed_machines[failure_types].sum()  
```  
Dès lors, nous **recalculons la fréquence des pannes** uniquement sur ce sous-ensemble de machine comme nous pouvons le voir ci-dessus. Puis nous allons ensuite faire en sorte de reconnaitre les cas où une machine a échoué c'est à dire quand`Machine failure = 1` mais aucun type de panne spécifique n’est renseigné comme nous pouvosn le voir sur le code ci-dessous. Grâce à cela, nous allons pouvoir mettre en évidence des anomalies ou simplement un manques d’information dans les données collectées dans le dataset.  
  
```python  
no_specific_failure_count = (failed_machines[failure_types].sum(axis=1) == 0).sum()  
  
```  
Après lancement du code , nous remarquons que sur le diagramme de classes générés, une nouvelle catégorie apparaît bien : **"No Specific Failure"**, correspondant aux machines en panne sans type de panne identifié et concerne 9 cas.  
- La répartition des autres types de panne (**HDF, OSF, PWF, TWF, RNF**) reste **cohérente** avec l’analyse précédente.  
- Les pannes les plus fréquentes sont liées à la **dissipation thermique (HDF)**, à la **surcharge (OSF)** et à l’**alimentation électrique (PWF)**.  
- Enfin, la présence de la catégorie **"No Specific Failure"** peut indiquer des données incomplètes ou mal collectées comme expliqué précédemment, ce qui explique la necessité d'avoir des capteurs fiables et de bons diagnostics.  
  
  
## 4. Modèle sans rééquilibrage des données  
  
### Sélection des variables pour l'entraînement du modèle  
  
Afin de préparer la phase de modélisation, nous avons dentifié les colonnes utiles comme **variables d’entrée (features)** et **variables de sortie (target)**.  
  
Nous avons affiché les noms des colonnes du dataset ainsi que leurs types afin de sélectionner les variables pertinentes pour l’entraînement du modèle :  
  
```python  
print("Noms des colonnes dans le dataset avec leurs types données :\n")  
print(df.dtypes)  
```  
Pour entraîner le modèle de maintenance prédictive, les variables sélectionnées comme entrées sont : la température de l’air (`Air temperature [K]`), la température du processus (`Process temperature [K]`), la vitesse de rotation (`Rotational speed [rpm]`), le couple (`Torque [Nm]`) et l’usure de l’outil (`Tool wear [min]`). Ces variables proviennent directement de capteurs physiques installés sur la machine et décrivent précisément ses conditions de fonctionnement.  
  
Du côté des sorties , les colonnes utilisées sont : `Machine failure`, qui indique si la machine est tombée en panne ou non, `TWF`, `HDF`, `PWF`, `OSF` , permettant de prédire non seulement la survenue d’une panne, mais aussi son type exact. Nous avons fait le choix que la colonne `RNF`, qui représente les pannes aléatoires, ne soit pas prise en compte en raison de son manque d'apport explicatif au modèle.  
  
### Entraînement d’un modèle sans rééquilibrage des données  
---  
Dans cette section, nous avons entraîné un modèle de réseau de neurones directement sur les données originales, sans les modifier. Cela signifie que nous n’avons pas tenté de corriger le déséquilibre important entre les classes que nous avons observés dans la partie précédente  
Dès lors, nous avons pas encore utilisé de technique de rééquilibrage comme **SMOTE** ou la **pondération des classes**.  
  
---  
  
#### Séparation des données  
  
Avant d’entraîner le modèle, nous lui avons donné :  
- des **entrées** (`X`) : ce sont les données que le modèle utilise pour apprendre.  
- plusieurs **sorties** (`Y`) : ce sont les informations que le modèle doit apprendre à prédire.  
  
Dans notre cas, nous avons gardé uniquement les colonnes **mesurées par des capteurs** pour les entrées, et supprimé les colonnes non pertinentes tel que `Product ID`, `Type` ou le type de pannes comme nous pouvons le voir ci-dessous.  
  
```python  
X = df.drop(columns=['Product ID','Type','Machine failure','TWF', 'HDF', 'PWF', 'OSF', 'RNF']).values  
Y = df[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF']]  
```  
Nous avons ensuite divisé les données :  
  
- **80 % pour l’entraînement**, pour que le modèle apprenne sur un grand nombre d’exemples.  
- **20 % pour les tests**, afin de vérifier si le modèle est capable de faire des prédictions correctes sur des données jamais vues.  
```python  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  
```  
---  
#### Architecture du modèle  
  
Par la suite, nous avons utilisé un modèle très simple de réseau de neurones avec seulement deux couches.  
Ainsi, pour aborder notre problème de maintenance prédictive, nous avons choisi de construire un réseau de neurones simple (**MLP**) via l’API Keras de TensorFlow.  
L’objectif est de prédire le type de panne parmi plusieurs catégories.  
  
  
```python  
model = keras.Sequential([  
keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
keras.layers.Dense(5, activation='softmax')  
])  
```  
  
- La **première couche** contient **64 neurones**, avec une activation **ReLU**, elle permet de détecter des relations entre les entrées. Elle est suffisante pour capturer les relations non linéaires dans ce dataset relativement simple.  
Le nombre de neurones permet d'avoir un bon compromis qui assure la rapidité d'entrainement.  
La fonction ReLU (`relu`)** permet d'introduire la non-linéarité  
tout en évitant les problèmes de gradient nul.  
  
- La **dernière couche** contient **5 neurones** avec une activation **softmax**, car on cherche à classer chaque panne selon un type unique (**multiclasse**). La fonction softmax permet d'obtenir une distribution de probabilités sur les classes.  
  
```python  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
````  
  
- Nous avons fait le choix d'utiliser **l'optimiseur Adam** car il combine les avantages de deux autres méthodes d’optimisation : **RMSProp** et **momentum**. Grâce à cette combinaison, Adam est bien adapté à une grande variété de jeux de données, y compris ceux présentant du **bruit**.  
  
- Concernant la fonction de perte, nous avons utilisé la **binary crossentropy**. Bien que le problème implique plusieurs classes, il est possible que chaque étiquette de panne soit **indépendante des autres**, ce qui correspond à un scénario **multilabel**. Dans ce cas, la binary crossentropy est appropriée, car elle considère chaque sortie comme une classification binaire.  
En revanche, si les classes étaient mutuellement exclusives, nous aurions utilisé la fonction **categorical crossentropy**, utilisée avec une transformation des labels via `to_categorical()`.  
  
  
---  
  
#### Entraînement du modèle  
  
Une fois le modèle défini, nous l'avons entraîné sur 50 epochs avec une taille de batch de 32, en surveillant à la fois l’évolution de la précision (`accuracy`) sur les données d’entraînement et de validation.  
  
```python  
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))  
````  
- Le choix de **50 épochs** permet de laisser au modèle suffisamment de temps pour apprendre les motifs sous-jacents dans les données, sans aller dans l'excès.  
En effet, un nombre **trop faible d’épochs** pourrait conduire à un **sous-apprentissage**, c’est-à-dire que le modèle ne comprend pas bien les relations dans les données.  
À l’inverse, un nombre **trop élevé d’épochs** pourrait provoquer un **surapprentissage** : le modèle mémorise trop bien les exemples d'entraînement mais échoue à généraliser sur des données nouvelles.  
  
  
- Le **batch size** détermine combien d'exemples le modèle voit avant de mettre à jour ses poids.  
Nous avons choisi une taille de **32** afin qu'elle ne soit pas trop petite, ce qui éviterait un entraînement trop lent et qu'elle ne soit pas trop grande non plus, ce qui pourrait empêcher le modèle d’apprendre efficacement des détails.  
  
Cette valeur permet un bon compromis entre la **stabilité** de l'entraînement et la **rapidité** de convergence.  
  
#### Suivi de l’évolution de la précision  
  
Nous avons tracé l’évolution de la **précision** sur les ensembles d’entraînement et de validation, afin de visualiser l’apprentissage du modèle :  
  
```python  
plt.figure(figsize=(8, 5))  
plt.plot(history.history['accuracy'], label='Train Accuracy')  
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  
plt.xlabel('Epochs')  
plt.ylabel('Accuracy')  
plt.legend()  
plt.title('Model Accuracy Over Epochs')  
plt.show()  
```  
À la lecture de ce graphique, on observe des fluctuations de l’accuracy sur la validation, ce qui peut signaler une instabilité ou un comportement aléatoire du modèle. L’écart parfois important entre la courbe d’entraînement et celle de validation peut aussi indiquer un problème d'overfitting, ou encore une mauvaise généralisation sur des cas minoritaires (comme les pannes).  
  
Cela confirme ce que nous avons observé précédemment sur le déséquilibre des classes : le modèle a des difficultés à apprendre efficacement des représentations utiles pour les cas moins fréquents.  
#### Matrice de confusion  
  
Afin de mieux comprendre le comportement du modèle sur les différentes classes, nous avons généré une **matrice de confusion** permettant de visualiser, pour chaque type de panne, combien de fois le modèle a correctement prédit la classe (valeurs sur la diagonale) et combien de fois il s’est trompé (valeurs hors diagonale).  
``` python  
labels = ["No Failure", "TWF", "HDF", "PWF", "OSF"] # Définition des noms de classes  
  
# Générer la matrice de confusion  
cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))  
  
# Affichage de la matrice sous forme de carte thermique  
sns.heatmap(  
cm,  
annot=True, # Affiche les valeurs dans chaque case  
fmt='d', # Format entier  
cmap='Blues', # Palette de couleur bleue  
xticklabels=Y_test.columns,  
yticklabels=Y_test.columns  
)  
  
plt.title("Matrice de confusion")  
plt.show()  
  
```  
- `labels = [...]` : définit les étiquettes des classes que l'on souhaite afficher sur les axes de la matrice (ici, les 4 types de pannes retenus ainsi que "No Failure").  
  
- `np.argmax(Y_test, axis=1)` : convertit les sorties multilabel (sous forme de vecteurs) en classes uniques, en prenant l'indice de la probabilité maximale pour chaque observation réelle.  
  
- `confusion_matrix(...)` : calcule la matrice de confusion, qui compare les valeurs réelles (Y_test) avec les prédictions (Y_pred) du modèle.  
  
- `sns.heatmap(...)` : permet d’afficher cette matrice sous forme de carte thermique. Cette visualisation facilite l’interprétation :  
- `annot=True` permet d'afficher les valeurs numériques dans chaque case,  
- `cmap='Blues'` applique une échelle de couleurs allant du bleu clair au bleu foncé selon l’intensité,  
  
  
La matrice de confusion permet dans notre projet:  
  
- de vérifier si les pannes sont bien détectées (présence de valeurs significatives sur la diagonale),  
- d’identifier les classes mal reconnues (valeurs faibles ou nulles en dehors de "No Failure"),  
- de relativiser la valeur de l’accuracy globale, qui peut être biaisée en cas de forte prédominance d’une classe.  
  
Ainsi ,nous constatons que la grande majorité des cas correctement prédits appartiennent à la classe "No Failure", tandis que les autres types de panne sont très mal détectés, voire pas du tout. Cela confirme le biais du modèle en faveur de la classe majoritaire.  
Nous pouvons donc conclure que ce modèle n’est pas fiable pour détecter les machines à risque, ce qui est pourtant l’objectif principal d’un système de maintenance prédictive.  
  
Il est donc nécessaire d’envisager des méthodes de rééquilibrage.  
  
## 5.Modèle avec rééquilibrage des données  
  
### Rééquilibrage du dataset avec SMOTE  
  
Dans cette partie, nous avons cherché à **corriger le déséquilibre des classes** observé précédemment dans notre jeu de données. En effet, certaines classes de pannes sont très peu représentées par rapport à la classe "No Failure", ce qui peut biaiser l’apprentissage du modèle et empêcher la détection des cas rares.  
  
  
#### Choix de la méthode : SMOTE pour données multilabel  
  
Nous avons opté pour la méthode **SMOTE ** qui permet de générer des exemples pour les classes minoritaires à partir de leurs plus proches voisins, sans se contenter de dupliquer les données. C’est une méthode efficace pour rééquilibrer les classes sans suradapter le modèle.  
  
Notre problème étant de type **multilabel** comme expliqué précédemment, nous avons appliqué SMOTE séparément sur chaque colonne de la variable cible `Y_train`.  
Dès lors, chaque colonne (`Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`) est traitée indépendamment comme une classification binaire à équilibrer.  
  
  
  
```python  
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split  
import pandas as pd  
  
# Définition des variables  
X = df.drop(columns=['Product ID','Type','Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])  
Y = df[['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF']] # multilabel  
  
# Séparation des données en ensemble d'entraînement et de test  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  
  
# Application de SMOTE indépendamment sur chaque colonne de Y  
X_resampled = []  
Y_resampled = []  
  
for col in Y_train.columns:  
sm = SMOTE(random_state=42)  
X_res, y_res = sm.fit_resample(X_train, Y_train[col])  
X_resampled.append(X_res)  
Y_resampled.append(y_res)  
  
# Construction du nouvel ensemble rééchantillonné  
  
```  
### Architecture du modèle après rééquilibrage  
  
Après avoir appliqué SMOTE pour équilibrer les classes, nous avons conçu un réseau de neurones plus important afin de mieux détecter les pannes peu courantes, tout en limitant le surapprentissage.  
  
  
```python  
X_train_smote = X_resampled[0] # X est identique pour chaque SMOTE appliqué  
Y_train_smote = pd.DataFrame({col: Y_resampled[i] for i, col in enumerate(Y_train.columns)})  
  
model = Sequential([  
Dense(128, input_shape=(X_train_smote.shape[1],), activation='relu', kernel_regularizer=l2(0.001)),  
BatchNormalization(),  
Dropout(0.4),  
  
Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  
BatchNormalization(),  
Dropout(0.3),  
  
Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  
BatchNormalization(),  
Dropout(0.2),  
  
Dense(Y_train_smote.shape[1], activation='sigmoid')  
])  
  
model.compile(  
optimizer=Adam(learning_rate=0.0005),  
loss='binary_crossentropy',  
metrics=[BinaryAccuracy(name='binary_acc'), AUC(name='AUC')]  
)  
  
model.summary()  
```  
- Le modèle est structuré avec trois couches denses intermédiaires (**128**, **64**, puis **32 neurones**) permettant d’apprendre des représentations progressivement plus abstraites à partir des données d’entrée.  
  
- L’activation utilisée dans toutes les couches cachées est **ReLU**. Elle est adaptée à l’apprentissage rapide et évite les problèmes de gradient nul que peuvent poser les fonctions `sigmoid` ou `tanh`.  
  
- Chaque couche dense est suivie d’une **normalisation par batch** (`BatchNormalization`), qui permet de stabiliser l’apprentissage et de réduire la sensibilité à l’initialisation.  
  
- Un **Dropout** est appliqué après chaque normalisation pour introduire une régularisation supplémentaire. Les taux de dropout (**0.4**, **0.3**, **0.2**) diminuent progressivement afin de conserver davantage d’information au fur et à mesure de la "profondeur" du réseau.  
  
- Une **régularisation L2** (`kernel_regularizer=l2(0.001)`) est appliquée sur toutes les couches cachées pour pénaliser les poids trop élevés et éviter la complexité inutile du modèle.  
  
- La **couche de sortie** contient **5 neurones** (correspondant aux 5 étiquettes de sortie) avec une activation **sigmoid** ce qui permet de traiter indépendamment la présence ou l’absence de chaque type de panne, conformément à la nature **multilabel** du problème.  
  
- Le modèle est quant à lui compilé avec l’optimiseur **Adam**, bien adapté à des jeux de données bruités et présentant des dynamiques différentes. Le **taux d’apprentissage** est fixé à **0.0005** afin d’éviter les oscillations trop fortes dans la descente de gradient.  
  
- La **fonction de perte** utilisée est la `binary_crossentropy`, car elle convient aux problèmes **multilabel** où chaque sortie est indépendante et binaire.  
  
- Enfin, les **métriques** retenues sont `BinaryAccuracy` et `AUC`.  
- `BinaryAccuracy` permet d’évaluer correctement la précision pour chaque étiquette.  
- `AUC` permet de mesurer la capacité du modèle à discriminer correctement entre les classes, même en cas de déséquilibre.  
  
Par conséquent ce nouveau modèle d'architecture intègre des mécanismes de régularisation et d’adaptation au déséquilibre, tout en étant conçue pour la classification multilabel dans un contexte industriel.  
  
### Entraînement du modèle  
  
Après avoir défini notre architecture de réseau de neurones, nous avons entraîné le modèle sur 100 epochs avec une taille de batch de 128, tout en réservant 20 % des données d'entraînement pour la validation.  
```python  
# === Entraînement ===  
#early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)  
  
history = model.fit(  
X_train_smote, Y_train_smote,  
validation_split=0.2,  
epochs=100,  
batch_size=128,  
verbose=1  
)  
  
# === Évaluation ===  
Y_pred = model.predict(X_test)  
Y_pred_bin = (Y_pred > 0.5).astype(int)  
  
print("\n=== Classification Report (par label) ===")  
print(classification_report(Y_test, Y_pred_bin, target_names=Y.columns))  
  
# === Affichage courbes loss / accuracy (optionnel) ===  
import matplotlib.pyplot as plt  
  
plt.figure()  
plt.plot(history.history['loss'], label='Train Loss')  
plt.plot(history.history['val_loss'], label='Val Loss')  
plt.title('Loss over epochs')  
plt.legend()  
plt.show()  
  
plt.figure()  
plt.plot(history.history['binary_acc'], label='Train Binary Accuracy')  
plt.plot(history.history['val_binary_acc'], label='Val Binary Accuracy')  
plt.title('Accuracy over epochs')  
plt.legend()  
plt.show()  
```  
  
- Le modèle a été entraîné sur le jeu de données rééquilibré (`X_train_smote`, `Y_train_smote`) afin de pallier le biais observé précédemment en faveur de la classe majoritaire ("No Failure").  
- La fonction `model.fit()` permet de lancer l'entraînement, et la variable `history` enregistre l’évolution des métriques (perte, précision) au fil des époques.  
- La prédiction sur les données de test (`X_test`) est ensuite binarisée à l’aide d’un seuil de 0.5, puis comparée aux vraies étiquettes `Y_test` via un rapport de classification.  
- Le nombre d’epochs a été fixé à 100 afin de permettre au modèle de converger correctement. Ce choix laisse suffisamment de temps au réseau pour apprendre des représentations pertinentes.  
  
- La taille de batch a été quant à elle fixée à 128. Cette valeur permet à la fois efficacité et stabilité . De plus, 128 permet des mises à jour régulières tout en conservant une certaine capacité de généralisation.  
  
- Nous avons également utilisé une séparation de validation `validation_split=0.2`, ce qui signifie que 20 % des données d’entraînement sont utilisées à chaque epoch pour évaluer la performance sur des données non vues. Cela permet de suivre l’évolution de l’apprentissage à travers les courbes de perte et de précision.  
  
  
Ainsi nous remarquons que la **courbe de Loss** montre une baisse régulière de la loss sur l’ensemble d'entraînement, avec des variations plus instables sur la validation. Cela reste attendu dans un cadre multilabel avec un dataset rééquilibré artificiellement. Les valeurs globales restent stables, indiquant un bon apprentissage.  
  
Par ailleurs, sur la **courbe de précision binaire (Binary Accuracy)** : la précision de validation dépasse 90 % sur la majorité des epochs, malgré des fluctuations. La précision sur l'entraînement progresse de façon stable et atteint également des valeurs élevées, ce qui témoigne d'une bonne généralisation globale du modèle.  
  
Après l’entraînement du modèle, nous avons évalué ses performances sur l’ensemble de test à l’aide d’une matrice de confusion multilabel. L’objectif était d’observer comment le modèle se comporte pour la prédiction de chaque type de panne, ainsi que pour la classe "No Failure".  
  
### Matrice de confusion  
  
```python  
labels = ["No Failure", "TWF", "HDF", "PWF", "OSF"]  
# Générer la matrice de confusion unique  
  
cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred_bin, axis=1))  
  
# Afficher la matrice de confusion sous forme de heatmap  
# Affichage de la matrice  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels = Y_test.columns,yticklabels=Y_test.columns)  
plt.title("Matrice de confusion")  
plt.show()  
  
print("Occurrences dans Y_test:")  
print(Y_test.sum())  
  
print("\nOccurrences dans Y_pred_bin:")  
print(pd.DataFrame(Y_pred_bin, columns=Y.columns).sum())  
  
model.save("modeleAI4I2020.")  
# Sauvegarder les entrées de validation  
np.save("X_val.npy", X_test)  
  
# Sauvegarder les vraies sorties de validation  
np.save("Y_val.npy", Y_test)  
```  
  
- Nous avons utilisé `np.argmax()` pour convertir les prédictions et les vraies valeurs en classes uniques.  
- La fonction `confusion_matrix()` de `sklearn` permet de construire la matrice croisant les valeurs réelles avec les valeurs prédites.  
- Nous avons ensuite affiché cette matrice avec `sns.heatmap()` en ajoutant les étiquettes de colonnes et lignes pour faciliter l’analyse visuelle.  
ur la matrice de confusion affichée, seules certaines classes comme "No Failure" et "TWF" sont visibles, ce qui peut laisser croire que le modèle ignore les autres classes. En réalité, il s’agit probablement d’un problème d’affichage lié à la bibliothèque utilisée, car les autres classes sont bien présentes dans les résultats.  
  
En consultant les tableaux récapitulatifs situés juste sous la matrice :  
  
- Dans `Y_test`, on observe bien 61 machines en panne, réparties entre :  
- 11 TWF  
- 17 HDF  
- 20 PWF  
- 18 OSF  
- Dans `Y_pred_bin`, on voit que le modèle prédit également des cas pour chaque type de panne :  
- 118 TWF  
- 119 HDF  
- 115 PWF  
- 117 OSF  
  
Ces résultats confirment que le modèle, après équilibrage des classes avec SMOTE et amélioration de son architecture, parvient désormais à détecter l’ensemble des classes, y compris les plus rares.  
On remarque donc une bonne évolution par rapport au modèle initial qui ne prédisait que l 'absence de panne.  
  
Dès lors, l’utilisation combinée de SMOTE et d’un réseau de neurones profond a donc permis d’améliorer les performances de détection des classes minoritaires ce qui aura un impact positif en milieu industriel dans le cadre de notre projet.  
  
Inférence embarquée sur microcontrôleur STM32L4R9  
  
## 5. Inférence embarquée sur microcontrôleur STM32L4R9  
  
Dans cette dernière phase du projet, notre objectif a été de déployer un modèle de réseau de neurones entraîné dans un jupyter notebook sur un microcontrôleur **STM32L4R9**, afin de permettre **l'inférence locale en temps réel** pour la maintenance prédictive.  
  
Cette approche s’inscrit dans une vision **Industrie 4.0**, où les capteurs, les unités de calcul embarquées (Uc), et les systèmes IA interagissent dans un écosystème connecté et intelligent.  
  
### Architecture globale du système  
  
![Schéma industrie 4.0](https://raw.githubusercontent.com/FabienSaliba/Embedded_AI/main/images/industrie4.0.png)
  
Comme illustré ci-dessus, chaque machine (M1, M2, M3) est équipée de capteurs physiques (température, vibrations, couple...) et d’un microcontrôleur.  
  
- Les capteurs envoient leurs mesures au microcontrôleur local (Uc).  
- Les microcontrôleurs peuvent transmettre les données à un serveur IA centralisé, qui analyse, stocke, entraîne et améliore le modèle (backpropagation).  
- Une des machines (M3) intègre directement un **petit RNN embarqué** entraîné préalablement, capable d’inférer en local pour détecter une panne.  
  
Ce petit modèle est mis à jour via un mécanisme d’apprentissage où chaque microcontrôleur collecte ses propres données, les remonte périodiquement au serveur, qui les utilise pour améliorer le modèle global avant de le redéployer dans les machines.  
  
---  
  
### Pipeline de développement et déploiement  
  
![Pipeline d’entraînement vers STM32](https://raw.githubusercontent.com/FabienSaliba/Embedded_AI/main/images/pipelinedentrainement.png)  
  
Le schéma ci-dessus résume le pipeline de traitement :  
  
- Le modèle est entraîné dans un **Jupyter Notebook** avec le dataset AI4I 2020.  
  
- Une fois validé, il est converti via **STM32Cube.AI** en code embarqué (.h5 → analyse → code C).  
  
- Ce code est intégré à un projet dans **STM32CubeIDE** pour générer l’application finale.  
  
- Un script Python permet ensuite de communiquer avec le microcontrôleur via **UART**, en envoyant les données et en récupérant les prédictions.  
  
  
Ce processus permet de passer d’un apprentissage Python pur à un modèle fonctionnel embarqué, exécuté localement sur STM32.  
  
---  
  
### Exemple de boucle d’inférence UART  
  
![Boucle UART AI420 STM32](https://raw.githubusercontent.com/FabienSaliba/Embedded_AI/main/images/boucleUART.png)  
  
Ce dernier schéma montre un cas d’usage basé sur le dataset ai420, mais totalement transposable à notre projet :  
  
- 784 pixels (float32) sont envoyés via UART2.  
- Le microcontrôleur exécute localement le réseau neuronal converti.  
- Il retourne les probabilités des 10 classes (float32) par UART.  
- Le script Python (`communication.py`) compare la classe prédite avec celle attendue, dans une boucle de test automatique.  
  
Dès lors, ce mécanisme est applicable à la maintenance prédictive . En effet, on envoie les **5 valeurs capteurs** à la STM32, le modèle embarqué retourne les **probabilités associées aux 5 types de panne** et script Python interroge le modèle et affiche les résultats (classe prédite).  
  
Ainsi grâce à cette architecture complète (collecte, entraînement, déploiement, inférence locale), nous pouvons réduire les temps de réaction aux pannes, assurer une bonne surveillance locale et adapter dynamiquement les modèles.  
  
### Intégration du modèle dans `app_x-cube-ai.c`  
  
Après avoir converti le modèle `.h5` en un modèle embarquable à l’aide de **STM32Cube.AI**, plusieurs étapes ont été nécessaires pour intégrer correctement le modèle dans le firmware du projet STM32.  
  
---  
  
#### Modifications dans les déclarations  
  
Dans le fichier `app_x-cube-ai.c`, nous avons défini plusieurs constantes utiles à la gestion du modèle et de la communication UART :  
  
```c  
#define BYTES_IN_FLOATS 5*4 // 5 entrées float (chaque float = 4 octets)  
#define TIMEOUT 1000  
#define SYNCHRONISATION 0xAB  
#define ACKNOWLEDGE 0xCD  
#define CLASS_NUMBER 5  
```  
- `BYTES_IN_FLOATS` représente la **taille totale des données d’entrée** envoyées à la carte STM32.  
Dans notre cas, le modèle prend en entrée **5 valeurs float32**, et chaque float est codé sur **4 octets**, soit un total de `5 x 4 = 20 octets` ce qui va de pair avec le modèle qu'on a décrit plus haut qui comprend 5 sorties.  
  
- `CLASS_NUMBER` correspond au **nombre de classes de sortie** du modèle.  
Comme notre réseau neuronal produit **5 probabilités de sortie** (une par type de panne), nous avons donc fixé `CLASS_NUMBER = 5`.  
  
- Les constantes `SYNCHRONISATION` et `ACKNOWLEDGE` sont utilisées pour **établir une communication UART fiable** entre le PC et la carte STM32.  
-  
Avant chaque inférence, la STM32 attend un signal de synchronisation (`SYNCHRONISATION = 0xAB`) et envoie en retour un accusé de réception (`ACKNOWLEDGE = 0xCD`), garantissant que les échanges de données sont correctement alignés.  
  
### Boucle principale d’inférence (UART)  
  
```c  
synchronize_UART();  
  
if (ai4i) {  
do {  
res = acquire_and_process_data(in_data);  
if (res == 0)  
res = ai_run();  
if (res == 0)  
res = post_process(out_data);  
} while (res == 0);  
}  
```  
  
La boucle principale d’inférence permet à la carte **STM32** de recevoir des données, de les traiter via le modèle de deep learning embarqué, puis de renvoyer les résultats de classification vers le PC.  
  
Dès lors, on commence par synchroniser la carte avec le PC via la fonction `synchronize_UART()`.  
Cette étape garantit que la STM32 n’exécute aucune inférence tant qu’elle n’a pas reçu un octet de synchronisation de la part du PC (valeur `0xAB`).  
  
Le bloc `if (ai4i)` permet de vérifier que le modèle est bien initialisé et prêt à l’emploi.  
`ai4i` est le nom de notre modèle modèle défini dans STM32Cube.AI lors de la génération. Il est aussi utilisé dans le code pour initialiser et exécuter l’inférence avec le bon réseau de neurones.  
  
Une fois la synchronisation validée, la boucle exécute les étapes suivantes :  
  
1. **Réception des données d’entrée** via `acquire_and_process_data()`  
→ Les **5 valeurs float** (soit 20 octets) sont reçues du PC et stockées dans le buffer d’entrée du modèle.  
  
2. **Exécution du modèle** avec la fonction `ai_run()`  
→ Elle réalise l’**inférence sur les données reçues**.  
  
3. **Envoi des résultats** avec `post_process()`  
→ Les sorties float32 sont **converties en `uint8_t` (entre 0 et 255)** et renvoyées au PC via UART.  
---  
### Erreur rencontrée  
  
Lors du lancement du script `communication.py`, une erreur est survenu dès la deuxième itération.  
  
Il s’agit d’un problème lié à l’utilisation de `np.argmax()` : ce bug peut apparaître lorsque les valeurs reçues via UART ne sont pas correctement interprétées. Cela peut être dû à une perte de synchronisation entre le PC et la STM32,  nous n'avons pas eu accès à une autre carte pour tester.
  
---  
  
### Conclusion  
  
A la suite de ce projet nous avons pu entraîner un modèle de classification multiclasses, le convertir via STM32Cube.AI, l’intégrer dans un firmware STM32, et tester en conditions réelles son exécution embarquée.  
  
Ce travail illustre un cas d’usage de machine learning embarqué en environnement industriel, avec une boucle complète depuis la collecte des données jusqu’à l’inférence autonome sur microcontrôleur.