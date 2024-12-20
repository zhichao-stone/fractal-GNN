import os
import json
import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def box_remove(F:nx.Graph, center_node, radius):
    nodes_to_remove = [node for node, distance in nx.single_source_shortest_path_length(F, center_node).items() if distance <= radius]
    F.remove_nodes_from(nodes_to_remove)

def box_number(F:nx.Graph, radius):
    steps = 0
    while len(F) > 0:
        max_degree_node = max(F.degree, key=lambda x: x[1])[0]
        box_remove(F, max_degree_node, radius)
        steps += 1
    return steps


def calculate_fractal(G:nx.MultiDiGraph, max_scale:int):
    x_vector = list(range(2, max_scale+1))
    G_y_vector = []
    ### 最大规模为graph直径的一半
    ### 统计一下graph的最长路径（以10为界）
    for i in range(max_scale-1):
        b = box_number(G.copy(), i + 2)
        G_y_vector.append(b)

    # 测定分形性，绘制log-log图
    logx = np.log(x_vector).reshape(-1, 1)
    G_logy = np.log(G_y_vector)

    coefficients = np.polyfit(logx.flatten(), G_logy, 1)
    slope, intercept = coefficients
    
    model = LinearRegression().fit(logx, G_logy)
    r_squared = model.score(logx, G_logy)
    G_logy_fit = slope * logx.flatten() + intercept

    return x_vector, G_y_vector, G_logy_fit, slope, intercept, r_squared


def test_fractal(G:nx.MultiDiGraph, data_title:str, draw_figure:bool=True, figure_dir:str=""):
    # 测定连通性
    components = nx.connected_components(nx.to_undirected(G.copy()))
    cc_nodes_num_list = [len(c) for c in sorted(components, key=len, reverse=True)]
    connected_num = len(cc_nodes_num_list)

    # 获得最大连通子图
    largest_cc = max(nx.connected_components(nx.to_undirected(G.copy())), key=len)
    lcc_G:nx.MultiDiGraph = G.subgraph(largest_cc).copy()
    diameter = nx.diameter(lcc_G)


    # 测定分形性，计算F分形维数
    can_test_fractal, max_scale = True, diameter//2
    if max_scale <= 2:
        can_test_fractal = False
        regression_1 = {"Can Test Fractality": False}
        regression_2 = {"Can Test Fractality": False}
    else:
        x_vector_1, G_y_vector_1, G_logy_fit_1, slope1, intercept1, r_squared1 = calculate_fractal(G, max_scale)
        regression_1 = {"Slope": slope1, "Intercept": intercept1, "R²": r_squared1}

        x_vector_2, G_y_vector_2, G_logy_fit_2, slope2, intercept2, r_squared2 = calculate_fractal(lcc_G, max_scale)
        regression_2 = {"Slope": slope2, "Intercept": intercept2, "R²": r_squared2}

    regression_result = {
        "Statistics of Graph": {
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Diameter": diameter,
        },
        "Linear Regression": {
            "Origin Graph": regression_1,
            "Largest Connected Component": regression_2
        },
        "Statistic of Connected Components": {
            "Number of CC": connected_num,
            "Nodes Number of each CC": cc_nodes_num_list
        }
    }

    if draw_figure and can_test_fractal:
        if figure_dir == "":
            figure_dir = os.path.join("./result", data_title)
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        plt.figure(figsize=(10, 6))
        plt.loglog(x_vector_1, G_y_vector_1, "o", label="Data points")
        plt.loglog(x_vector_1, np.exp(G_logy_fit_1), label=f"slope={slope1:.2f}\nR²={r_squared1:.2f}", linestyle="--")
        plt.xlabel("X (log scale)")
        plt.ylabel("Y (log scale)")
        plt.title("Log-Log Plot with Linear Regression")
        plt.savefig(os.path.join(figure_dir, f"logplot_{data_title}.png"))
        plt.clf()

        plt.figure(figsize=(10, 6))
        plt.loglog(x_vector_2, G_y_vector_2, "o", label="Data points")
        plt.loglog(x_vector_2, np.exp(G_logy_fit_2), label=f"slope={slope2:.2f}\nR²={r_squared2:.2f}", linestyle="--")
        plt.xlabel("X (log scale)")
        plt.ylabel("Y (log scale)")
        plt.title("Log-Log Plot with Linear Regression")
        plt.savefig(os.path.join(figure_dir, f"logplot_{data_title}_largest_cc.png"))
        plt.clf()

    return regression_result
    


if __name__ == "__main__":
    from loading import *
    from utils import dump_json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="citeseer")
    args = parser.parse_args()
    DATA = str(args.data).lower()

    if DATA in NODE_PRED_DATA or DATA in LINK_PRED_DATA:
        load_data = LOAD_FUNCTION_MAP[DATA]()
        g = load_data[0]
        G = dgl.to_networkx(g)

        regression_result = test_fractal(G, DATA, draw_figure=True, figure_dir=f"./results/{DATA}")

        dump_json(regression_result, f"results/linear_regression_{DATA}.json")

    elif DATA in GRAPH_PRED_DATA:
        from tqdm import tqdm
        graphs, labels, num_classes = LOAD_FUNCTION_MAP[DATA]()
        print(f"number of graphs: {len(graphs)} , number of classes: {num_classes}")
        regression_results = []
        index = 0
        for g in tqdm(list(graphs)):
            G = dgl.to_networkx(g)
            result = test_fractal(G, f"{DATA}_{index}", draw_figure=True, figure_dir=f"./results/{DATA}/{index}")
            regression_results.append(result)
            index += 1

        dump_json(regression_results, f"results/linear_regression_{DATA}.json")
        