#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Maze Runner - jeu console en un seul fichier.
Conforme au cahier des charges :
- Génération de labyrinthe (DFS backtracking) avec option "braid"
- Contrôles WASD (et ZQSD) + commandes H (indice), R (régénérer), N (nouveau), Q (quitter)
- Rendu ASCII avec option brouillard (fog) et option emoji
- Chronomètre, comptage de pas, utilisation d'indices
- Calcul de score paramétrable (alpha, beta, gamma)
- Sauvegarde locale des meilleurs scores dans ~/.mazerunner_scores.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterable, Deque
from collections import deque

# -----------------------------
# Utilitaires et constantes
# -----------------------------

SCORES_PATH = os.path.expanduser("~/.mazerunner_scores.json")
SCORES_SCHEMA = 1

DIFF_PRESETS = {
    "easy": (15, 15),
    "normal": (25, 25),
    "hard": (35, 35),
    "insane": (51, 51),
}

# Map de caractères par défaut (ASCII)
ASCII_TILES = {
    "wall": "#",
    "floor": "·",
    "start": "S",
    "exit": "E",
    "player": "@",
    "unknown": " ",  # utilisé par le brouillard
    "hint": "*",     # surbrillance du prochain pas d'indice
}

# Map emoji (optionnel)
EMOJI_TILES = {
    "wall": "🧱",
    "floor": "·",
    "start": "S",
    "exit": "🚪",
    "player": "🙂",
    "unknown": " ",
    "hint": "⭐",
}

# -----------------------------
# Configuration & état du jeu
# -----------------------------

@dataclass
class Config:
    width: int
    height: int
    difficulty: str
    seed: Optional[int] = None
    fog: bool = False
    vision: int = 0  # rayon si fog=True
    emoji: bool = False
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 50.0
    braid: float = 0.0
    debug: bool = False

    def tiles(self):
        return EMOJI_TILES if self.emoji else ASCII_TILES


@dataclass
class GameState:
    player_pos: Tuple[int, int]
    steps: int = 0
    hints_used: int = 0
    start_time: Optional[float] = None
    ended: bool = False
    last_hint_cell: Optional[Tuple[int, int]] = None

    def start_timer_if_needed(self):
        if self.start_time is None:
            self.start_time = time.time()

    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# -----------------------------
# Génération du labyrinthe
# -----------------------------

class Maze:
    """
    Représente un labyrinthe à base de grille (1 = mur, 0 = sol).
    Les indices des cellules "sol" sont impairs pour width/height impairs.
    """
    def __init__(self, width: int, height: int):
        # Validation taille impaire
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1

        # Grille initiale pleine de murs
        self.grid: List[List[int]] = [[1 for _ in range(self.width)] for _ in range(self.height)]
        self.start: Tuple[int, int] = (1, 1)   # position départ (sol)
        self.exit: Tuple[int, int] = (self.width - 2, self.height - 2)  # position sortie (sera mise à jour)

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == 1

    def set_floor(self, x: int, y: int):
        self.grid[y][x] = 0

    def neighbors_cells(self, cx: int, cy: int) -> List[Tuple[int, int, int, int]]:
        """
        Retourne une liste de voisins candidats depuis une cellule (indices impairs),
        sous forme (nx, ny, wx, wy) où (nx,ny) est la cellule prochaine, (wx,wy) le mur entre les deux.
        """
        dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        out = []
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            wx, wy = cx + dx // 2, cy + dy // 2
            if self.is_in_bounds(nx, ny):
                out.append((nx, ny, wx, wy))
        random.shuffle(out)
        return out

    def generate(self, seed: Optional[int] = None, debug: bool = False, braid: float = 0.0):
        """
        Génération par DFS backtracking.
        - seed : reproductibilité
        - braid : proportion de culs-de-sac à ouvrir pour réduire la linéarité (0..1)
        """
        if seed is not None:
            random.seed(seed)

        # 1) Initialisation
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = 1  # tout mur

        # 2) DFS depuis (1,1)
        stack = [(1, 1)]
        self.set_floor(1, 1)

        visited = set([(1, 1)])

        while stack:
            cx, cy = stack[-1]
            # Chercher voisins non visités (cellules impaires)
            candidates = []
            for nx, ny, wx, wy in self.neighbors_cells(cx, cy):
                if (nx, ny) not in visited and self.is_in_bounds(nx, ny):
                    candidates.append((nx, ny, wx, wy))

            if candidates:
                nx, ny, wx, wy = random.choice(candidates)
                # Ouvrir le mur entre les deux cellules + la cellule cible
                self.set_floor(wx, wy)
                self.set_floor(nx, ny)
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # 3) Déterminer la sortie comme la cellule accessible la plus éloignée de start
        #    (distance BFS sur le graphe de cellules “sol”)
        dist, farthest = self._bfs_distance_from(self.start)
        self.exit = farthest

        # 4) Option : "braid" - diminuer le nombre de culs-de-sac
        if braid > 0.0:
            self._apply_braid(braid)

        if debug:
            dead_ends = self._count_dead_ends()
            print(f"[DEBUG] Labyrinthe généré {self.width}x{self.height} | culs-de-sac: {dead_ends} | sortie: {self.exit}", file=sys.stderr)

    def _bfs_distance_from(self, src: Tuple[int, int]) -> Tuple[List[List[int]], Tuple[int, int]]:
        """
        BFS sur la grille de sols pour obtenir distances et la cellule la plus éloignée.
        Retourne (matrice distance, cellule la plus éloignée).
        """
        dist = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        q: Deque[Tuple[int, int]] = deque()
        sx, sy = src
        dist[sy][sx] = 0
        q.append((sx, sy))
        farthest = src

        while q:
            x, y = q.popleft()
            if dist[y][x] > dist[farthest[1]][farthest[0]]:
                farthest = (x, y)
            for nx, ny in self._passable_neighbors(x, y):
                if dist[ny][nx] == -1:
                    dist[ny][nx] = dist[y][x] + 1
                    q.append((nx, ny))
        return dist, farthest

    def _passable_neighbors(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if self.is_in_bounds(nx, ny) and not self.is_wall(nx, ny):
                yield (nx, ny)

    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        return list(self._passable_neighbors(x, y))

    def to_ascii(self) -> List[str]:
        """
        Retourne une représentation brute ASCII (sans joueur ni HUD).
        """
        rows = []
        for y in range(self.height):
            rows.append("".join(ASCII_TILES["wall"] if self.grid[y][x] else ASCII_TILES["floor"] for x in range(self.width)))
        return rows

    def _count_dead_ends(self) -> int:
        """
        Compte les culs-de-sac (cases sol avec 1 seul voisin accessible).
        """
        count = 0
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if not self.is_wall(x, y):
                    deg = sum(1 for _ in self._passable_neighbors(x, y))
                    if deg == 1:
                        count += 1
        return count

    def _apply_braid(self, p: float):
        """
        Ouvre certains culs-de-sac en fonction de la probabilité p (0..1),
        en supprimant un mur vers une case voisine non accessible (création de boucle).
        """
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if not self.is_wall(x, y):
                    deg = sum(1 for _ in self._passable_neighbors(x, y))
                    if deg == 1 and random.random() < p:
                        # Essayer d'ouvrir un mur parmi les 4 directions pour créer une seconde sortie
                        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                        random.shuffle(dirs)
                        for dx, dy in dirs:
                            nx, ny = x+dx, y+dy
                            if self.is_in_bounds(nx, ny) and self.is_wall(nx, ny):
                                # S'assurer que derrière ce mur il y a un sol (de l'autre côté)
                                bx, by = nx+dx, ny+dy
                                if self.is_in_bounds(bx, by) and not self.is_wall(bx, by):
                                    # Ouvrir le mur
                                    self.set_floor(nx, ny)
                                    break


# -----------------------------
# Pathfinder (indices)
# -----------------------------

class Pathfinder:
    """
    Implémente un BFS pour trouver le plus court chemin et renvoyer le PROCHAIN pas à faire.
    """
    @staticmethod
    def bfs_next_step(maze: Maze, src: Tuple[int,int], dst: Tuple[int,int]) -> Optional[Tuple[int,int]]:
        if src == dst:
            return None
        q: Deque[Tuple[int,int]] = deque([src])
        prev = {src: None}
        while q:
            x, y = q.popleft()
            if (x, y) == dst:
                break
            for nx, ny in maze.neighbors(x, y):
                if (nx, ny) not in prev:
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))
        if dst not in prev:
            return None  # pas de chemin
        # Reconstituer depuis dst jusqu'à src, puis retourner le premier pas après src
        cur = dst
        path = []
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path[1] if len(path) >= 2 else None


# -----------------------------
# Rendu console
# -----------------------------

class Renderer:
    """
    Rendu ASCII simple, avec HUD.
    Le brouillard (fog) masque les cases hors rayon de vision (Chebyshev) si activé.
    """
    def render(self, maze: Maze, state: GameState, config: Config) -> None:
        self._clear()
        tiles = config.tiles()
        # Construire les lignes à afficher
        lines: List[str] = []

        # HUD (haut)
        hud = self._hud_line(state, config)
        lines.append(hud)

        # Corps du labyrinthe
        for y in range(maze.height):
            row_chars = []
            for x in range(maze.width):
                ch = tiles["wall"] if maze.is_wall(x, y) else tiles["floor"]

                # Placer S/E
                if (x, y) == maze.start:
                    ch = tiles["start"]
                if (x, y) == maze.exit:
                    ch = tiles["exit"]

                # Brouillard
                if config.fog:
                    if not self._is_visible(state.player_pos, (x, y), config.vision):
                        ch = tiles["unknown"]

                # Indice (surbrillance ponctuelle d'une case)
                if state.last_hint_cell and (x, y) == state.last_hint_cell:
                    ch = tiles["hint"]

                # Joueur par-dessus tout
                if (x, y) == state.player_pos:
                    ch = tiles["player"]

                row_chars.append(ch)
            lines.append("".join(row_chars))

        # HUD (bas)
        lines.append(hud)

        # Affichage
        print("\n".join(lines))

    def render_summary(self, maze: Maze, state: GameState, config: Config, score: int) -> None:
        """
        Affiche un récapitulatif de fin.
        """
        print("\n=== FIN DE PARTIE ===")
        print(f"Diff: {config.difficulty} | Taille: {maze.width}x{maze.height} | Seed: {config.seed}")
        print(f"Temps: {state.elapsed():.1f}s | Pas: {state.steps} | Indices: {state.hints_used}")
        print(f"Score: {score}")

    def _hud_line(self, state: GameState, config: Config) -> str:
        t = state.elapsed()
        return f"[Tps: {t:5.1f}s] [Pas: {state.steps:4d}] [Indices: {state.hints_used:2d}] [Diff: {config.difficulty}]"

    @staticmethod
    def _is_visible(p: Tuple[int,int], cell: Tuple[int,int], radius: int) -> bool:
        # Distance de Chebyshev (vision carrée) : max(|dx|,|dy|) <= radius
        (px, py), (cx, cy) = p, cell
        return max(abs(px - cx), abs(py - cy)) <= radius

    @staticmethod
    def _clear():
        # Effacement simple, compatible Windows/macOS/Linux
        os.system("cls" if os.name == "nt" else "clear")


# -----------------------------
# Scoreboard (JSON local)
# -----------------------------

class ScoreBoard:
    def __init__(self, path: str = SCORES_PATH):
        self.path = path
        self.data = {"schema": SCORES_SCHEMA, "entries": []}  # type: ignore

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            # Validation minimale du schéma
            if self.data.get("schema") != SCORES_SCHEMA:
                self.data = {"schema": SCORES_SCHEMA, "entries": []}
        except Exception:
            # En cas d'erreur d'accès/parse, on redémarre sans persistance
            self.data = {"schema": SCORES_SCHEMA, "entries": []}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            # Non bloquant: on ignore l'échec d'écriture
            pass

    def submit(self, score: int, meta: dict):
        entry = {
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "score": score,
        }
        entry.update(meta)
        self.data["entries"].append(entry)
        self.save()

    def top(self, n: int = 10, key: Optional[dict] = None):
        """
        Retourne le top n filtré par un dictionnaire "key" (ex: difficulté+taille)
        trié par score décroissant.
        """
        entries = self.data.get("entries", [])
        if key:
            def match(e): return all(e.get(k) == v for k, v in key.items())
            entries = list(filter(match, entries))
        return sorted(entries, key=lambda e: e.get("score", 0), reverse=True)[:n]


# -----------------------------
# Calcul du score
# -----------------------------

def compute_score(maze: Maze, state: GameState, cfg: Config) -> int:
    """
    score = max(0, base - temps_s*alpha - pas*beta - indices*gamma)
    où base = width*height
    """
    base = maze.width * maze.height
    t = state.elapsed()
    raw = base - t * cfg.alpha - state.steps * cfg.beta - state.hints_used * cfg.gamma
    return max(0, int(round(raw)))


# -----------------------------
# Entrées utilisateur
# -----------------------------

MOVE_KEYS = {
    # WASD
    "w": (0, -1),
    "a": (-1, 0),
    "s": (0, 1),
    "d": (1, 0),
    # ZQSD (claviers AZERTY)
    "z": (0, -1),
    "q": (-1, 0),
}

def read_command() -> str:
    """
    Lecture ligne par ligne d’une commande utilisateur.
    Retourne la chaîne en minuscule sans espaces.
    """
    try:
        cmd = input("> ").strip().lower()
        return cmd
    except (EOFError, KeyboardInterrupt):
        return "q"


# -----------------------------
# Boucle de jeu
# -----------------------------

def run_game(cfg: Config):
    # Instanciation
    maze = Maze(cfg.width, cfg.height)
    maze.generate(seed=cfg.seed, debug=cfg.debug, braid=cfg.braid)
    state = GameState(player_pos=maze.start)
    renderer = Renderer()

    scoreboard = ScoreBoard()
    scoreboard.load()

    # Clé de classement
    key_meta = {
        "difficulty": cfg.difficulty,
        "size": f"{maze.width}x{maze.height}",
        "seed": cfg.seed,
    }

    # Rendu initial
    renderer.render(maze, state, cfg)

    # Boucle principale
    while True:
        cmd = read_command()
        state.last_hint_cell = None  # l'indication visuelle d'indice est temporaire (un rendu)
        if cmd == "":
            # Pas d'action
            renderer.render(maze, state, cfg)
            continue

        # Quitter
        if cmd == "q":
            if not state.ended and state.start_time is not None:
                # Confirmation sommaire
                print("Quitter ? (o/N)")
                if input().strip().lower() != "o":
                    renderer.render(maze, state, cfg)
                    continue
            break

        # Indice
        if cmd == "h":
            nxt = Pathfinder.bfs_next_step(maze, state.player_pos, maze.exit)
            if nxt is not None:
                state.last_hint_cell = nxt
                state.hints_used += 1
            renderer.render(maze, state, cfg)
            continue

        # Régénérer même graine
        if cmd == "r":
            maze.generate(seed=cfg.seed, debug=cfg.debug, braid=cfg.braid)
            state = GameState(player_pos=maze.start)
            renderer.render(maze, state, cfg)
            continue

        # Nouveau labyrinthe (nouvelle graine)
        if cmd == "n":
            # Nouvelle graine aléatoire
            cfg.seed = random.randint(0, 2**31 - 1)
            maze.generate(seed=cfg.seed, debug=cfg.debug, braid=cfg.braid)
            state = GameState(player_pos=maze.start)
            # mettre à jour la clé de classement
            key_meta["seed"] = cfg.seed
            renderer.render(maze, state, cfg)
            continue

        # Mouvement : WASD / ZQSD, ou tenter de parser "up/down/left/right"
        moved = False
        if cmd in MOVE_KEYS:
            dx, dy = MOVE_KEYS[cmd]
            moved = try_move(maze, state, dx, dy)
        elif cmd in ("up", "down", "left", "right"):
            mapping = {
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0),
            }
            dx, dy = mapping[cmd]
            moved = try_move(maze, state, dx, dy)
        else:
            # Aide minimale
            if cmd in ("?", "help"):
                print("Commandes: W/A/S/D ou Z/Q/S/D pour bouger, H=indice, R=régénérer, N=nouveau, Q=quitter")
                time.sleep(1.2)

        # Vérifier arrivée
        if state.player_pos == maze.exit and not state.ended:
            state.ended = True
            # Calcul du score + affichage résumé
            score = compute_score(maze, state, cfg)
            renderer.render(maze, state, cfg)
            renderer.render_summary(maze, state, cfg, score)
            # Enregistrer au classement
            meta = dict(key_meta)
            meta.update({
                "time_s": round(state.elapsed(), 2),
                "steps": state.steps,
                "hints": state.hints_used,
            })
            scoreboard.submit(score, meta)

            # Afficher top 5 local pour ce contexte
            print("\nTop 5 local (même diff/taille/seed) :")
            top = scoreboard.top(5, key=key_meta)
            for i, e in enumerate(top, 1):
                print(f"{i:2d}. {e.get('score',0):5d} | {e.get('when','')} | t={e.get('time_s','?')}s, pas={e.get('steps','?')}, ind={e.get('hints','?')}")

            # Proposer de rejouer
            print("\nRejouer ? (O/n)")
            if input().strip().lower() not in ("n", "non"):
                # Rejouer même config, nouvelle graine
                cfg.seed = random.randint(0, 2**31 - 1)
                key_meta["seed"] = cfg.seed
                maze.generate(seed=cfg.seed, debug=cfg.debug, braid=cfg.braid)
                state = GameState(player_pos=maze.start)
                renderer.render(maze, state, cfg)
                continue
            else:
                break

        # Rendu après action
        renderer.render(maze, state, cfg)

    # Sortie propre
    print("Au revoir !")


def try_move(maze: Maze, state: GameState, dx: int, dy: int) -> bool:
    """
    Tente de déplacer le joueur de (dx,dy). Renvoie True si déplacement effectué.
    Démarre le chrono au premier déplacement.
    """
    px, py = state.player_pos
    nx, ny = px + dx, py + dy
    if maze.is_in_bounds(nx, ny) and not maze.is_wall(nx, ny):
        state.start_timer_if_needed()
        state.player_pos = (nx, ny)
        state.steps += 1
        return True
    return False


# -----------------------------
# Parsing des arguments
# -----------------------------

def parse_args(argv: List[str]) -> Config:
    parser = argparse.ArgumentParser(
        description="Maze Runner - labyrinthe console en un seul fichier (ASCII)."
    )

    parser.add_argument("--difficulty", choices=DIFF_PRESETS.keys(), default="normal",
                        help="Préréglage de difficulté (défaut: normal)")
    parser.add_argument("--width", type=int, default=None, help="Largeur du labyrinthe (impair ≥ 7)")
    parser.add_argument("--height", type=int, default=None, help="Hauteur du labyrinthe (impair ≥ 7)")
    parser.add_argument("--seed", type=int, default=None, help="Graine aléatoire pour reproductibilité")
    parser.add_argument("--fog", action="store_true", help="Active le brouillard de guerre")
    parser.add_argument("--vision", type=int, default=4, help="Rayon de vision si --fog (défaut: 4)")
    parser.add_argument("--emoji", action="store_true", help="Rendu avec emoji au lieu d'ASCII")
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient temps pour le score (défaut: 1.0)")
    parser.add_argument("--beta", type=float, default=0.5, help="Coefficient pas pour le score (défaut: 0.5)")
    parser.add_argument("--gamma", type=float, default=50.0, help="Pénalité par indice pour le score (défaut: 50)")
    parser.add_argument("--braid", type=float, default=0.0, help="Taux d'ouverture des culs-de-sac (0..1)")
    parser.add_argument("--debug", action="store_true", help="Affiche quelques informations de debug")

    args = parser.parse_args(argv)

    # Déterminer taille selon difficulté si non spécifiée
    if args.width is None or args.height is None:
        w, h = DIFF_PRESETS[args.difficulty]
    else:
        w, h = args.width, args.height

    # Validation des tailles
    w = int(w)
    h = int(h)
    if w < 7 or h < 7:
        print("Erreur: width/height doivent être ≥ 7.", file=sys.stderr)
        sys.exit(2)
    if w % 2 == 0 or h % 2 == 0:
        # On corrige silencieusement en arrondissant au supérieur impair
        w = w + (1 if w % 2 == 0 else 0)
        h = h + (1 if h % 2 == 0 else 0)

    # Validation vision
    if args.fog and args.vision < 1:
        print("Erreur: --vision doit être ≥ 1 si --fog est activé.", file=sys.stderr)
        sys.exit(2)

    # Validation braid
    if not (0.0 <= args.braid <= 1.0):
        print("Erreur: --braid doit être dans [0,1].", file=sys.stderr)
        sys.exit(2)

    cfg = Config(
        width=w,
        height=h,
        difficulty=args.difficulty,
        seed=args.seed,
        fog=args.fog,
        vision=args.vision if args.fog else 0,
        emoji=args.emoji,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        braid=args.braid,
        debug=args.debug,
    )
    return cfg


# -----------------------------
# Point d’entrée
# -----------------------------

def main():
    cfg = parse_args(sys.argv[1:])
    if cfg.debug:
        print(f"[DEBUG] Config: {cfg}", file=sys.stderr)
    run_game(cfg)


if __name__ == "__main__":
    main()