# NLP-Llama4-Scratch

Ce projet vise à entraîner un modèle de langage de type Llama 4 à partir de zéro, en utilisant PyTorch. Le code inclus implémente les composants clés d'un modèle Transformer, y compris l'attention multi-tête, l'incorporation positionnelle rotative (RoPE), et une architecture Mixture of Experts (MoE).

## Structure du Projet

Le projet est organisé comme suit :

* `NLP - Llama 4 - from scratch.ipynb`: Notebook Jupyter contenant le code principal pour l'entraînement du modèle.

## Dépendances

Les bibliothèques Python suivantes sont requises pour exécuter le code :

* torch
* torch.nn
* torch.nn.functional
* torch.optim
* math
* os
* collections
* re
* datasets
* matplotlib.pyplot
* unicodedata

Vous pouvez installer les dépendances nécessaires en utilisant pip :

\`\`\`bash
pip install torch datasets matplotlib
\`\`\`

## Configuration

Le notebook commence par une classe `Config` qui centralise tous les hyperparamètres et les paramètres architecturaux du modèle. Ceux-ci incluent :

* Taille du modèle (`d_model`)
* Nombre de couches (`n_layers`)
* Nombre de têtes d'attention (`n_heads`)
* Taille du bloc (`block_size`)
* Paramètres pour la normalisation RMS et RoPE
* Taille du vocabulaire (`vocab_size`)
* Paramètres MoE (nombre d'experts locaux, nombre d'experts par token)
* Paramètres d'entraînement (taux d'apprentissage, taille du lot, nombre d'époques, intervalle d'évaluation)

La classe `Config` calcule également les valeurs dérivées comme la dimension de la clé/requête (`d_k`) et les tailles des couches intermédiaires pour les experts MoE.

## Tokenizer

Une classe simple `SimpleCharTokenizer` est implémentée pour convertir le texte en séquences d'ID de tokens et vice versa. Le tokenizer est initialisé avec un corpus de texte, construit un vocabulaire de caractères uniques et crée des mappings caractère-entier et entier-caractère.

## Chargement et Préparation des Données

La fonction `load_and_prepare_data` charge le dataset Wikitext en français, crée un tokenizer et prépare les données d'entraînement. Le texte est normalisé, encodé en séquences de tokens, et divisé en séquences de longueur fixe (`block_size`) avec des masques d'attention. Si une séquence est plus courte que `block_size`, elle est complétée par du padding.

## Composants du Modèle

Le code implémente les composants clés suivants d'un modèle Transformer :

* `RMSNorm`: Applique la normalisation Root Mean Square à un tenseur.
* `RotaryPositionalEmbedding`: Calcule et applique l'incorporation positionnelle rotative (RoPE) aux tenseurs de requêtes et de clés dans l'attention.
* `Attention`: Implémente l'attention multi-tête avec RoPE et masquage causal.
* `ExpertMLP`: Implémente un réseau perceptron multicouche pour un expert dans l'architecture MoE.
* `MoE`: Implémente la couche Mixture of Experts, qui route les tokens vers les experts les plus appropriés.
* `TransformerBlock`: Combine l'attention multi-tête, MoE et les connexions résiduelles en un bloc Transformer.
* `LlamaModel`: Assemble les blocs Transformer, l'incorporation et la couche de sortie pour former le modèle Llama complet.

## Entraînement

Le notebook inclut une boucle d'entraînement qui itère sur les données, effectue une propagation avant, calcule la perte, effectue une rétropropagation et met à jour les paramètres du modèle. Il inclut également l'évaluation périodique du modèle sur un ensemble de validation et la sauvegarde des checkpoints du modèle.

## Chargement et Inférence du Modèle

Le notebook inclut également le code pour charger un modèle entraîné à partir d'un checkpoint et générer du texte.

## Exécution du Code

Pour exécuter le code, ouvrez le notebook `NLP - Llama 4 - from scratch.ipynb` dans un environnement Jupyter et exécutez les cellules séquentiellement. Assurez-vous d'avoir toutes les dépendances nécessaires installées.

## Notes

* Le code est conçu pour être exécuté sur un GPU si disponible.
* La taille du lot et le nombre d'époques peuvent être ajustés en fonction de vos ressources et de vos besoins.
* Le modèle peut être entraîné sur un plus grand corpus de texte pour de meilleures performances.
* L'architecture MoE peut être personnalisée en ajustant le nombre d'experts et le nombre d'experts par token.
