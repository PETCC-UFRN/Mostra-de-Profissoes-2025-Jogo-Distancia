import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import BeautifyIcon
from streamlit_folium import st_folium
import heapq
import numpy as np
from collections import deque

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Jogo de Rotas", layout="wide")
st.title("🗺️ Jogo de Rotas: Você vs. o Algoritmo")

# --- FUNÇÕES E CLASSES DO ALGORITMO (sem alterações) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def graph(lat_o, lon_o, lat_d, lon_d, mode="walk", margin=500):
    dist = haversine(lat_o, lon_o, lat_d, lon_d) + margin
    lat_c = (lat_o + lat_d)/2
    lon_c = (lon_o + lon_d)/2
    return ox.graph_from_point(center_point=(lat_c, lon_c), dist=dist, network_type=mode, simplify=True)

def heuristic_fn(G, u, v):
    node_u = G.nodes[u]
    node_v = G.nodes[v]
    return haversine(node_u['y'], node_u['x'], node_v['y'], node_v['x'])

class Problem:
    def __init__(self, initial_state, actions, transition_model, goal_test, step_cost):
        self.initial_state = initial_state
        self.actions = actions
        self.transition_model = transition_model
        self.goal_test = goal_test
        self.step_cost = step_cost

class Node:
    def __init__(self, problem, parent=None, action=None):
        self.parent = parent
        self.action = action
        if parent is None:
            self.state = problem.initial_state
            self.path_cost = 0.0
        else:
            self.state = problem.transition_model(parent.state, action)
            self.path_cost = parent.path_cost + problem.step_cost(parent.state, action)
    def __eq__(self, other): return isinstance(other, Node) and self.state == other.state
    def __hash__(self): return hash(self.state)
    def __lt__(self, other): return self.path_cost < other.path_cost

class AStarSearch:
    def __init__(self, problem):
        self.problem = problem
        self.frontier = []
        self.explored = set()

    def search(self):
        root = Node(problem=self.problem)
        f_value = root.path_cost + self.problem.heuristic(self.problem.G, root.state, self.problem.goal_node)
        heapq.heappush(self.frontier, (f_value, root))
        while self.frontier:
            _, node = heapq.heappop(self.frontier)
            if self.problem.goal_test(node.state): return node
            if node.state in self.explored: continue
            self.explored.add(node.state)
            for action in self.problem.actions(node.state):
                child = Node(self.problem, node, action)
                if child.state not in self.explored:
                    f_child = child.path_cost + self.problem.heuristic(self.problem.G, child.state, self.problem.goal_node)
                    heapq.heappush(self.frontier, (f_child, child))
        return None

class MapProblem(Problem):
    def __init__(self, G, start_node, goal_node):
        self.G = G; self.goal_node = goal_node
        super().__init__(initial_state=start_node, actions=self.actions_fn, transition_model=self.transition_fn, goal_test=self.goal_test_fn, step_cost=self.step_cost_fn)
    def actions_fn(self, state): return list(self.G.neighbors(state))
    def transition_fn(self, state, action): return action
    def goal_test_fn(self, state): return state == self.goal_node
    def step_cost_fn(self, state, action): return 1.0

class AStarMapProblem(MapProblem):
    def __init__(self, G, start_node, goal_node, heuristic_fn):
        super().__init__(G, start_node, goal_node)
        self.heuristic = heuristic_fn
    def step_cost_fn(self, state, action):
        return self.G.get_edge_data(state, action)[0]['length']

# --- FUNÇÕES AUXILIARES ---
@st.cache_data
def get_graph(lat_o, lon_o, lat_d, lon_d, mode, margin):
    return graph(lat_o, lon_o, lat_d, lon_d, mode=mode, margin=margin)

def calculate_astar_path(G, start_node, goal_node):
    problem = AStarMapProblem(G, start_node=start_node, goal_node=goal_node, heuristic_fn=heuristic_fn)
    search = AStarSearch(problem)
    solution_node = search.search()
    if not solution_node: return None, float('inf')
    path = []; node = solution_node
    while node: path.append(node.state); node = node.parent
    path.reverse()
    return path, solution_node.path_cost

def initialize_session_state():
    if "game_started" not in st.session_state: st.session_state.game_started = False
    if "map_center" not in st.session_state: st.session_state.map_center = None
    if "map_zoom" not in st.session_state: st.session_state.map_zoom = 15

# --- INTERFACE (SIDEBAR) ---
with st.sidebar:
    st.header("1. Defina a Rota")
    lat_origin = st.number_input("Latitude Origem", value=-5.8318, format="%.4f")
    lon_origin = st.number_input("Longitude Origem", value=-35.2055, format="%.4f")
    lat_goal = st.number_input("Latitude Destino", value=-5.8414, format="%.4f")
    lon_goal = st.number_input("Longitude Destino", value=-35.1971, format="%.4f")
    
    if st.button("Iniciar Jogo", type="primary"):
        with st.spinner("Calculando a melhor rota..."):
            G = get_graph(lat_origin, lon_origin, lat_goal, lon_goal, mode="walk", margin=200)
            start_node = ox.nearest_nodes(G, lon_origin, lat_origin)
            goal_node = ox.nearest_nodes(G, lon_goal, lat_goal)
            astar_path, astar_dist = calculate_astar_path(G, start_node, goal_node)
            
            st.session_state.graph = G
            st.session_state.start_node, st.session_state.goal_node = start_node, goal_node
            st.session_state.astar_path, st.session_state.astar_dist = astar_path, astar_dist
            st.session_state.user_path = [start_node]
            st.session_state.map_center = [(lat_origin + lat_goal)/2, (lon_origin + lon_goal)/2]
            st.session_state.map_zoom = 16
            st.session_state.game_started = True
            st.rerun()

# --- LÓGICA PRINCIPAL ---
initialize_session_state()

if not st.session_state.game_started:
    st.info("Defina os pontos de origem e destino na barra lateral e clique em 'Iniciar Jogo'.")
else:
    G = st.session_state.graph
    start_node, goal_node = st.session_state.start_node, st.session_state.goal_node
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("2. Seu Jogo")
        st.write("Clique nos pontos azuis para avançar. Para voltar, clique no ponto pulsante.")
        
        user_dist = 0
        if len(st.session_state.user_path) > 1:
            try:
                user_dist = nx.path_weight(G, st.session_state.user_path, weight='length')
            except (nx.NetworkXNoPath, nx.NodeNotFound): pass
        
        st.metric("Distância da Rota Ótima (A*)", f"{st.session_state.astar_dist:.2f} m")
        st.metric("Sua Distância Atual", f"{user_dist:.2f} m")

        if goal_node in st.session_state.user_path:
            st.success("Você chegou ao destino!")
            diferenca = user_dist - st.session_state.astar_dist
            st.info(f"Sua rota foi {diferenca:.2f} m mais longa que a ótima.")
        
        if st.button("Reiniciar Meu Caminho"):
            st.session_state.user_path = [start_node]
            st.rerun()
            
    with col2:
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles="cartodbpositron")

        current_node = st.session_state.user_path[-1]
        valid_next_nodes = list(G.neighbors(current_node))

        if goal_node in st.session_state.user_path:
            path_coords_astar = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in st.session_state.astar_path]
            # Usando uma linha tracejada para diferenciar bem
            folium.PolyLine(
                path_coords_astar,
                color="lime",
                weight=7,
                opacity=0.9,
                dash_array='10, 5',
                tooltip="Rota Ótima (Algoritmo)"
            ).add_to(m)

        # Desenha a linha e os pontos do caminho do usuário
        if len(st.session_state.user_path) > 1:
            path_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in st.session_state.user_path]
            folium.PolyLine(path_coords, color="#00ffff", weight=5, opacity=0.8, tooltip="Sua Rota").add_to(m)
        
        # Desenha os pontos já visitados (exceto o último)
        for node_id in st.session_state.user_path[:-1]:
            node_data = G.nodes[node_id]
            folium.CircleMarker(location=(node_data['y'], node_data['x']), radius=5, color="#00ffff", fill_color="#00ffff", fill=True).add_to(m)

        # Destaca o ponto ATUAL com um ícone pulsante para a função "voltar"
        node_data = G.nodes[current_node]
        icon = BeautifyIcon(icon_shape='marker', border_color='#00ffff', background_color='#00ffff', inner_icon_style='font-size:0px;', spin=True)
        folium.Marker(location=(node_data['y'], node_data['x']), icon=icon, tooltip="Sua posição atual (Clique para voltar)").add_to(m)

        # Desenha os próximos passos válidos
        for node_id in valid_next_nodes:
            if node_id not in st.session_state.user_path:
                node_data = G.nodes[node_id]
                folium.CircleMarker(location=(node_data['y'], node_data['x']), radius=7, color="#0000ff", fill=True, fill_color="#0000ff", tooltip=f"Avançar para o nó {node_id}").add_to(m)

        # Marcadores de início e fim
        folium.Marker([G.nodes[start_node]['y'], G.nodes[start_node]['x']], popup="Origem", icon=folium.Icon(color="green", icon="play")).add_to(m)
        folium.Marker([G.nodes[goal_node]['y'], G.nodes[goal_node]['x']], popup="Destino", icon=folium.Icon(color="red", icon="stop")).add_to(m)
        
        # Exibe o mapa
        map_data = st_folium(m, width='100%', height=500)

        # Salva o estado do mapa de forma segura
        if map_data and map_data.get("center") and map_data.get("zoom"):
            st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
            st.session_state.map_zoom = map_data["zoom"]

        # Lógica de clique corrigida
        clicked_node = None
        click_coords = None

        if map_data and map_data.get("last_object_clicked"):
            click_coords = map_data["last_object_clicked"]
        elif map_data and map_data.get("last_clicked"):
            click_coords = map_data["last_clicked"]

        if click_coords:
            clicked_node = ox.nearest_nodes(G, click_coords['lng'], click_coords['lat'])

            if clicked_node == current_node and len(st.session_state.user_path) > 1:
                st.session_state.user_path.pop()
                st.rerun()
            elif clicked_node in valid_next_nodes and clicked_node not in st.session_state.user_path:
                st.session_state.user_path.append(clicked_node)
                st.rerun()