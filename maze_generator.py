import numpy as np
import random
import os

def generate_maze(width=21, height=21, extra_paths=0):
    """
    Gera um labirinto usando Randomized DFS e, opcionalmente,
    adiciona caminhos extras para criar loops e soluções múltiplas.
    """
    # Garante que as dimensões sejam ímpares para criar paredes
    width = width if width % 2 != 0 else width + 1
    height = height if height % 2 != 0 else height + 1

    # Inicia o labirinto com todas as células como paredes
    maze = np.ones((height, width), dtype=np.uint8)
    
    # --- Passo 1: Gerar o labirinto perfeito (lógica original) ---
    stack = []
    start_x, start_y = random.randrange(1, width, 2), random.randrange(1, height, 2)
    maze[start_y, start_x] = 0
    stack.append((start_x, start_y))

    while stack:
        current_x, current_y = stack[-1]
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = current_x + dx, current_y + dy
            if 0 < nx < width -1 and 0 < ny < height -1 and maze[ny, nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            next_x, next_y = random.choice(neighbors)
            wall_x, wall_y = (current_x + next_x) // 2, (current_y + next_y) // 2
            maze[wall_y, wall_x] = 0
            maze[next_y, next_x] = 0
            stack.append((next_x, next_y))
        else:
            stack.pop()
            
    # --- Passo 2: Adicionar caminhos extras (NOVA LÓGICA) ---
    walls_removed = 0
    while walls_removed < extra_paths:
        # Tenta remover uma parede interna aleatória
        # Escolhe coordenadas aleatórias, evitando as bordas externas
        r = random.randint(1, height - 2)
        c = random.randint(1, width - 2)
        
        # Se for uma parede, transforma em caminho
        if maze[r, c] == 1:
            maze[r, c] = 0
            walls_removed += 1

    # --- Passo 3: Definir entrada e saída (lógica original) ---
    maze[1, 0] = 0
    maze[height - 2, width - 1] = 0

    return maze

if __name__ == "__main__":
    MAZE_DIR = "mazes"
    if not os.path.exists(MAZE_DIR):
        os.makedirs(MAZE_DIR)
        print(f"Pasta '{MAZE_DIR}' criada.")

    num_mazes = int(input("Quantos labirintos deseja gerar? "))
    width = int(input("Qual a largura desejada (ímpar)? "))
    height = int(input("Qual a altura desejada (ímpar)? "))
    
    # NOVO INPUT DO USUÁRIO
    extra_paths = int(input("Quantos caminhos extras (atalhos) você quer adicionar? (ex: 5-15) "))

    for i in range(num_mazes):
        print(f"Gerando labirinto {i + 1}...")
        maze_data = generate_maze(width, height, extra_paths=extra_paths)
        filename = os.path.join(MAZE_DIR, f"maze_{width}x{height}_{i+1}.txt")
        np.savetxt(filename, maze_data, fmt='%d')
        print(f"Labirinto salvo em '{filename}'")

    print("\nGeração concluída!")