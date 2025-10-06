import streamlit as st
import numpy as np
import os
import heapq
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Jogo do Labirinto", layout="wide")
st.title("🧩 Jogo do Labirinto: Você vs. A*")

# --- LÓGICA DO ALGORITMO A* (Inalterada) ---
def heuristic_manhattan(a, b):
    (x1, y1) = a; (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(maze, start, goal):
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal: break
        (r, c) = current
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (r + dr, c + dc)
            if 0 <= neighbor[0] < maze.shape[0] and 0 <= neighbor[1] < maze.shape[1] and maze[neighbor[0], neighbor[1]] == 0:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic_manhattan(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
    path = []
    if goal in came_from:
        node = goal
        while node != start: path.append(node); node = came_from[node]
        path.append(start)
        path.reverse()
    return path

# --- NOVAS FUNÇÕES DE VISUALIZAÇÃO COM PILLOW ---

# Constantes para o desenho
CELL_SIZE = 25
COLORS = {
    "parede": "#111111",
    "caminho": "#FFFFFF",
    "usuario": "#00FFFF",
    "astar": "#39FF14",
    "inicio": "#00FF00",
    "fim": "#FF0000"
}

def draw_maze(maze, user_path, astar_path=None):
    """Desenha o labirinto como um objeto de imagem da biblioteca Pillow."""
    height, width = maze.shape
    img_width = width * CELL_SIZE
    img_height = height * CELL_SIZE

    img = Image.new("RGB", (img_width, img_height), color=COLORS["caminho"])
    draw = ImageDraw.Draw(img)

    display_grid = np.full(maze.shape, "caminho", dtype=object)
    display_grid[maze == 1] = "parede"
    
    if astar_path:
        for r, c in astar_path:
            display_grid[r, c] = "astar"
    if user_path:
        for r, c in user_path:
            display_grid[r, c] = "usuario"
            
    display_grid[st.session_state.start_node] = "inicio"
    display_grid[st.session_state.goal_node] = "fim"

    for r in range(height):
        for c in range(width):
            color = COLORS[display_grid[r, c]]
            draw.rectangle(
                [(c * CELL_SIZE, r * CELL_SIZE), 
                 ((c + 1) * CELL_SIZE - 1, (r + 1) * CELL_SIZE - 1)],
                fill=color
            )
    return img

# --- FUNÇÕES AUXILIARES (Inalteradas) ---
@st.cache_data
def load_maze(filepath):
    return np.loadtxt(filepath, dtype=np.uint8)

def get_available_mazes(directory="mazes"):
    if not os.path.exists(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")]

# --- INICIALIZAÇÃO E SIDEBAR (Inalterado) ---
if "game_started" not in st.session_state:
    st.session_state.game_started = False

with st.sidebar:
    st.header("1. Configure o Jogo")
    available_mazes = get_available_mazes()
    if not available_mazes:
        st.warning("Nenhum labirinto encontrado! Execute `maze_generator.py` primeiro.")
    else:
        maze_file = st.selectbox("Escolha um labirinto:", [os.path.basename(f) for f in available_mazes])
        if st.button("Iniciar Jogo", type="primary"):
            maze_path = os.path.join("mazes", maze_file)
            maze_data = load_maze(maze_path)
            st.session_state.maze = maze_data
            st.session_state.start_node = (1, 0)
            st.session_state.goal_node = (maze_data.shape[0] - 2, maze_data.shape[1] - 1)
            with st.spinner("Calculando a melhor rota..."):
                astar_path = a_star_search(maze_data, st.session_state.start_node, st.session_state.goal_node)
            st.session_state.astar_path = astar_path
            st.session_state.user_path = [st.session_state.start_node]
            st.session_state.game_started = True
            st.rerun()

# --- LÓGICA PRINCIPAL DO JOGO (Adaptada) ---
if not st.session_state.game_started:
    st.info("Escolha um labirinto e clique em 'Iniciar Jogo' na barra lateral.")
else:
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.header("2. Seu Jogo")
        st.write("Clique em uma célula adjacente para se mover.")
        user_dist = len(st.session_state.user_path) - 1
        astar_dist = len(st.session_state.astar_path) - 1 if st.session_state.astar_path else 0
        st.metric("Distância da Rota Ótima (A*)", f"{astar_dist} passos")
        st.metric("Sua Distância Atual", f"{user_dist} passos")
        
        current_node = st.session_state.user_path[-1]
        if current_node == st.session_state.goal_node:
            st.success("🎉 Você chegou ao destino!")
            diferenca = user_dist - astar_dist
            if diferenca == 0: st.info("Você fez a rota perfeita!")
            else: st.warning(f"Sua rota foi {diferenca} passo(s) mais longa que a ótima.")
        
        if st.button("Reiniciar Meu Caminho"):
            st.session_state.user_path = [st.session_state.start_node]
            st.rerun()
            
        show_astar = st.toggle("Mostrar caminho do A*", value=False)
        
    with col2:
        image = draw_maze(
            st.session_state.maze,
            st.session_state.user_path,
            st.session_state.astar_path if show_astar else None
        )
        # Exibe a imagem e captura as coordenadas do clique
        value = streamlit_image_coordinates(image, key="maze_click")

        if value:
            # Converte as coordenadas de pixel para coordenadas da grade
            clicked_col = value['x'] // CELL_SIZE
            clicked_row = value['y'] // CELL_SIZE
            clicked_node = (clicked_row, clicked_col)
            
            current_node = st.session_state.user_path[-1]

            # Verifica se o clique foi dentro dos limites da grade
            if 0 <= clicked_row < st.session_state.maze.shape[0] and 0 <= clicked_col < st.session_state.maze.shape[1]:
                # Verifica se o movimento é válido (adjacente e não é parede)
                if heuristic_manhattan(current_node, clicked_node) == 1 and st.session_state.maze[clicked_node] == 0:
                    if clicked_node in st.session_state.user_path:
                        idx = st.session_state.user_path.index(clicked_node)
                        st.session_state.user_path = st.session_state.user_path[:idx+1]
                    else:
                        st.session_state.user_path.append(clicked_node)
                    st.rerun()