# Maze Runner en Python

**Maze Runner** génère un labyrinthe procédural en ASCII (ou en emoji en option). Le joueur commence au point **S** et doit atteindre la sortie **E** en déplaçant son personnage `@`.  

Caractéristiques principales :
- Génération aléatoire avec option de **seed** pour la reproductibilité.  
- Différents niveaux de difficulté : *easy*, *normal*, *hard*, *insane*.  
- Commandes simples : **W/A/S/D** ou **Z/Q/S/D**.  
- Système de **brouillard de guerre** avec rayon de vision paramétrable.  
- Indices (basés sur BFS) pour révéler le prochain pas du chemin optimal.  
- Calcul de **score** prenant en compte le temps, le nombre de pas et l’usage d’indices.  
- Sauvegarde locale des meilleurs scores au format JSON.  
- Une option *braid* pour réduire le nombre de culs-de-sac.  

---

## ⚙️ Installation

Cloner le dépôt :

```bash
git clone https://github.com/P1erreD/Maze-Runner.git
cd Maze-Runner
````

Aucune dépendance externe n’est nécessaire, le jeu utilise uniquement la bibliothèque standard de Python (≥ 3.9).

---

## 🚀 Utilisation

Lancer le jeu :

```bash
python mazerunner.py
```

Options disponibles :

```bash
python mazerunner.py --help
```

Exemples :

```bash
# Labyrinthe 25x25 en mode normal
python mazerunner.py --difficulty normal

# Labyrinthe avec seed fixée
python mazerunner.py --difficulty hard --seed 12345

# Mode brouillard, rayon 4, rendu emoji
python mazerunner.py --fog --vision 4 --emoji
```

---

## ⌨️ Commandes

* `W/A/S/D` ou `Z/Q/S/D` : déplacer le joueur.
* `H` : afficher un indice (pénalité de score).
* `R` : régénérer le labyrinthe avec la même seed.
* `N` : générer un nouveau labyrinthe avec une seed aléatoire.
* `Q` : quitter la partie.
* `?` ou `help` : afficher l’aide des commandes.

---

## 🏆 Système de score

Formule :

```
score = max(0, base - temps*α - pas*β - indices*γ)
base = largeur * hauteur
```

* `α` (par défaut 1.0) : coefficient du temps (secondes).
* `β` (par défaut 0.5) : coefficient des pas.
* `γ` (par défaut 50) : pénalité par indice utilisé.

Le classement est sauvegardé localement dans `~/.mazerunner_scores.json`.

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**.
Voir le fichier [LICENSE](LICENSE) pour plus de détails.
