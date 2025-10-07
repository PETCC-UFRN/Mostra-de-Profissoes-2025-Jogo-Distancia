import streamlit as st
import numpy as np
import os
import heapq
import json
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONSTANTES ---
RANKING_FILE = "ranking.json"

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Jogo do Labirinto", layout="wide")
st.title("🧩 Jogo do Labirinto: Você vs. A*")

# --- FUNÇÕES DE RANKING ---

def load_ranking():
    """Carrega o ranking do arquivo JSON. Retorna um dicionário vazio se o arquivo não existir."""
    if not os.path.exists(RANKING_FILE):
        return {}
    try:
        with open(RANKING_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_ranking(data):
    """Salva o dicionário de ranking no arquivo JSON."""
    with open(RANKING_FILE, "w") as f:
        json.dump(data, f, indent=4)

def add_score(maze_file, player_name, score):
    """Adiciona uma nova pontuação ao ranking, mantém a lista ordenada e salva."""
    rankings = load_ranking()
    maze_key = os.path.basename(maze_file)
    
    # Garante que o nome seja maiúsculo e tenha no máximo 6 caracteres
    player_name = player_name.strip().upper()[:6]
    if not player_name:
        st.error("O nome não pode estar vazio!")
        return

    new_score = {"name": player_name, "score": score}

    if maze_key not in rankings:
        rankings[maze_key] = []

    rankings[maze_key].append(new_score)
    
    # Ordena a lista de scores para a fase atual (menor score é melhor)
    rankings[maze_key].sort(key=lambda x: x["score"])
    
    # Mantém apenas o top 10
    rankings[maze_key] = rankings[maze_key][:10]

    save_ranking(rankings)
    st.toast(f"Pontuação de {player_name} salva!", icon="🏆")

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

# --- FUNÇÕES DE VISUALIZAÇÃO COM PILLOW (Inalterada) ---
CELL_SIZE = 25
COLORS = {
    "parede": "#111111", "caminho": "#FFFFFF", "usuario": "#00FFFF",
    "astar": "#39FF14", "inicio": "#00FF00", "fim": "#FF0000"
}

def draw_maze(maze, user_path, astar_path=None):
    height, width = maze.shape
    img_width, img_height = width * CELL_SIZE, height * CELL_SIZE
    img = Image.new("RGB", (img_width, img_height), color=COLORS["caminho"])
    draw = ImageDraw.Draw(img)
    display_grid = np.full(maze.shape, "caminho", dtype=object)
    display_grid[maze == 1] = "parede"
    if astar_path:
        for r, c in astar_path: display_grid[r, c] = "astar"
    if user_path:
        for r, c in user_path: display_grid[r, c] = "usuario"
    display_grid[st.session_state.start_node] = "inicio"
    display_grid[st.session_state.goal_node] = "fim"
    for r in range(height):
        for c in range(width):
            color = COLORS[display_grid[r, c]]
            draw.rectangle(
                [(c * CELL_SIZE, r * CELL_SIZE), ((c + 1) * CELL_SIZE - 1, (r + 1) * CELL_SIZE - 1)],
                fill=color
            )
    return img

# --- FUNÇÕES AUXILIARES (Inalteradas) ---
@st.cache_data
def load_maze(filepath):
    return np.loadtxt(filepath, dtype=np.uint8)

def get_available_mazes(directory="mazes"):
    if not os.path.exists(directory): return []
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt")])

# --- INICIALIZAÇÃO E SIDEBAR ---
if "game_started" not in st.session_state:
    st.session_state.game_started = False

with st.sidebar:
    st.header("1. Configure o Jogo")
    available_mazes = get_available_mazes()
    if not available_mazes:
        st.warning("Nenhum labirinto encontrado! Execute `maze_generator.py` primeiro.")
    else:
        maze_file_path = st.selectbox("Escolha um labirinto:", available_mazes, format_func=os.path.basename)
        
        if st.button("Iniciar Jogo", type="primary"):
            maze_data = load_maze(maze_file_path)
            st.session_state.maze = maze_data
            st.session_state.maze_file = maze_file_path
            st.session_state.start_node = (1, 0)
            st.session_state.goal_node = (maze_data.shape[0] - 2, maze_data.shape[1] - 1)
            with st.spinner("Calculando a melhor rota..."):
                astar_path = a_star_search(maze_data, st.session_state.start_node, st.session_state.goal_node)
            st.session_state.astar_path = astar_path
            st.session_state.user_path = [st.session_state.start_node]
            st.session_state.score_saved = False # Reseta o estado de pontuação
            st.session_state.game_started = True
            st.rerun()

    # --- Exibição do Ranking na Sidebar ---
    if st.session_state.game_started:
        st.divider()
        st.header("🏆 Ranking da Fase")
        rankings = load_ranking()
        maze_key = os.path.basename(st.session_state.maze_file)
        scores = rankings.get(maze_key, [])
        if not scores:
            st.info("Seja o primeiro a pontuar!")
        else:
            df = pd.DataFrame(scores)
            df.index = df.index + 1
            df.columns = ["Nome", "Passos"]
            st.dataframe(df, use_container_width=True)

# --- LÓGICA PRINCIPAL DO JOGO ---
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
        
        # --- Lógica de Fim de Jogo e Salvamento de Score ---
        if current_node == st.session_state.goal_node:
            st.success("🎉 Você chegou ao destino!")
            diferenca = user_dist - astar_dist
            if diferenca == 0: st.info("Você fez a rota perfeita!")
            else: st.warning(f"Sua rota foi {diferenca} passo(s) mais longa que a ótima.")

            if not st.session_state.get('score_saved', False):
                with st.form("ranking_form"):
                    player_name = st.text_input("Seu nome (até 8 letras):", max_chars=8, placeholder="JOGADOR")
                    submitted = st.form_submit_button("Salvar Pontuação")
                    if submitted and player_name:
                        add_score(st.session_state.maze_file, player_name, user_dist)
                        st.session_state.score_saved = True
                        st.rerun()
            else:
                st.info("Sua pontuação para esta partida já foi salva!")

        if st.button("Reiniciar Meu Caminho"):
            st.session_state.user_path = [st.session_state.start_node]
            st.session_state.score_saved = False # Permite salvar de novo se chegar ao fim
            st.rerun()
            
        show_astar = st.toggle("Mostrar caminho do A*", value=False)
        
    with col2:
        image = draw_maze(
            st.session_state.maze,
            st.session_state.user_path,
            st.session_state.astar_path if show_astar else None
        )
        value = streamlit_image_coordinates(image, key="maze_click")

        if value and current_node != st.session_state.goal_node:
            clicked_col = value['x'] // CELL_SIZE
            clicked_row = value['y'] // CELL_SIZE
            clicked_node = (clicked_row, clicked_col)
            
            if 0 <= clicked_row < st.session_state.maze.shape[0] and 0 <= clicked_col < st.session_state.maze.shape[1]:
                if heuristic_manhattan(current_node, clicked_node) == 1 and st.session_state.maze[clicked_node] == 0:
                    if clicked_node in st.session_state.user_path:
                        idx = st.session_state.user_path.index(clicked_node)
                        st.session_state.user_path = st.session_state.user_path[:idx+1]
                    else:
                        st.session_state.user_path.append(clicked_node)
                    st.rerun()
