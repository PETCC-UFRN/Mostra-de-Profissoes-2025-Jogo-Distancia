import numpy as np
import random
import os

def generate_maze(width=21, height=21):
    """Gera um labirinto usando Randomized Depth-First Search."""
    # Garante que as dimensões sejam ímpares para criar paredes
    width = width if width % 2 != 0 else width + 1
    height = height if height % 2 != 0 else height + 1

    # Inicia o labirinto com todas as células como paredes
    maze = np.ones((height, width), dtype=np.uint8)
    
    # Pilha para o algoritmo DFS
    stack = []

    # Escolhe um ponto de partida aleatório (deve ser uma célula de caminho, não parede)
    start_x, start_y = random.randrange(1, width, 2), random.randrange(1, height, 2)
    maze[start_y, start_x] = 0  # 0 representa um caminho
    stack.append((start_x, start_y))

    while stack:
        current_x, current_y = stack[-1]
        
        # Lista de vizinhos não visitados
        neighbors = []
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = current_x + dx, current_y + dy
            if 0 < nx < width -1 and 0 < ny < height -1 and maze[ny, nx] == 1:
                neighbors.append((nx, ny))

        if neighbors:
            # Escolhe um vizinho aleatório
            next_x, next_y = random.choice(neighbors)
            
            # Remove a parede entre a célula atual e a escolhida
            wall_x, wall_y = (current_x + next_x) // 2, (current_y + next_y) // 2
            maze[wall_y, wall_x] = 0
            
            # Marca o vizinho como caminho e o adiciona à pilha
            maze[next_y, next_x] = 0
            stack.append((next_x, next_y))
        else:
            # Se não houver vizinhos não visitados, volta (backtrack)
            stack.pop()
            
    # Define a entrada e a saída do labirinto
    maze[1, 0] = 0  # Entrada
    maze[height - 2, width - 1] = 0 # Saída

    return maze

if __name__ == "__main__":
    MAZE_DIR = "mazes"
    if not os.path.exists(MAZE_DIR):
        os.makedirs(MAZE_DIR)
        print(f"Pasta '{MAZE_DIR}' criada.")

    num_mazes = int(input("Quantos labirintos deseja gerar? "))
    width = int(input("Qual a largura desejada (ímpar)? "))
    height = int(input("Qual a altura desejada (ímpar)? "))

    for i in range(num_mazes):
        print(f"Gerando labirinto {i + 1}...")
        maze_data = generate_maze(width, height)
        filename = os.path.join(MAZE_DIR, f"maze_{width}x{height}_{i+1}.txt")
        np.savetxt(filename, maze_data, fmt='%d')
        print(f"Labirinto salvo em '{filename}'")

    print("\nGeração concluída!")
