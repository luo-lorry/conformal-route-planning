from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GraphConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.transforms import LineGraph, RandomNodeSplit
from torch_geometric import seed_everything

seed_everything(1075)
node_rename = {node: id for id, node in enumerate(range(388, 934))}
nodefile = 'ChicagoSketch_node.tntp'
node = pd.read_csv(nodefile, sep='\t', usecols=['node', 'X', 'Y'])
flowfile = 'ChicagoSketch_flow.tntp'
colname = 'Volume '
flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

node['node'] = node['node'].map(node_rename)
node = node[node['node'].notna()]

flow['From '] = flow['From '].map(node_rename)
flow['To '] = flow['To '].map(node_rename)
flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
flow.drop(flow[flow[colname] <= 0].index, inplace=True)
flow[colname] = np.log(flow[colname])

scaler = StandardScaler()
node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)
# minmax = MinMaxScaler()
# flow[[colname]] = minmax.fit_transform(flow[[colname]].values)

df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(
    columns={'X': 'X1', 'Y': 'Y1'})
df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(
    columns={'X': 'X2', 'Y': 'Y2'})
df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
G = nx.from_pandas_edgelist(df2, source='s', target='r', edge_attr='w', create_using=nx.DiGraph())
airport = from_networkx(G)
airport.x = torch.from_numpy(node[['X', 'Y']].values).to(torch.float32)
print(airport)

G_line_graph = nx.line_graph(G, create_using=nx.DiGraph())
airport_line_graph = from_networkx(G_line_graph)
airport_line_graph.x = torch.from_numpy(np.vstack([edge_name_to_x[e] for e in G_line_graph.nodes])).to(torch.float32)
airport_line_graph.y = torch.from_numpy(np.vstack([edge_name_to_y[e] for e in G_line_graph.nodes])).to(torch.float32)
print(airport_line_graph)

split = RandomNodeSplit(num_val=0.1, num_test=0.4)
data = split(airport_line_graph)
data = data.to(device)

edge_array = np.array(list(dict(G_line_graph.nodes).keys()))
edge_index_train = edge_array[data.train_mask.cpu().numpy()]
edge_index_val = edge_array[data.val_mask.cpu().numpy()]
edge_index_calib_test = edge_array[data.test_mask.cpu().numpy()]

x_cqr = [[169, 171], [229, 231], [166, 49], [49, 168], [168, 169], [449, 451], [458, 457], [171, 243], [252, 250], [437, 439], [183, 246], [457, 443], [439, 449], [250, 437], [246, 247], [451, 459], [247, 252], [231, 166], [243, 183], [459, 458]]
x_cqr_new = [[169, 171], [229, 231], [166, 49], [49, 168], [168, 169], [449, 451], [458, 457], [171, 243], [252, 250], [437, 439], [183, 246], [457, 443], [439, 449], [250, 437], [246, 247], [451, 459], [247, 252], [231, 166], [243, 183], [459, 458]]
x_qr = [[321, 320], [200, 196], [229, 207], [320, 525], [325, 321], [324, 325], [196, 324], [204, 199], [29, 344], [344, 476], [476, 479], [208, 225], [207, 208], [225, 220], [71, 457], [479, 482], [457, 443], [199, 200], [220, 204], [484, 444], [482, 483], [483, 484], [525, 29], [444, 71]]
x_bl = [[321, 320], [200, 196], [229, 207], [320, 525], [325, 321], [324, 325], [196, 324], [204, 199], [29, 344], [344, 476], [476, 479], [208, 225], [207, 208], [225, 220], [71, 457], [479, 482], [457, 443], [199, 200], [220, 204], [484, 444], [482, 483], [483, 484], [525, 29], [444, 71]]
position = {n: (x, y) for n, x, y in node.values}

# 创建绘图对象和子图
fig, ax = plt.subplots(1, 4, figsize=(20, 8), gridspec_kw={'wspace': 0.1, 'hspace': 0})
# bl
nx.draw_networkx(G.to_undirected(), position, ax=ax[0], node_color='lightgray', node_size=10, edge_color='lightgray',
                 width=5, alpha=0.3, with_labels=False)
nx.draw_networkx_edges(G.to_undirected(), position, edgelist=x_bl, ax=ax[0], edge_color='blue', width=8)
# 获取高亮路线经过的节点列表
highlighted_nodes = list(set([edge[0] for edge in x_bl] + [edge[1] for edge in x_bl]))
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=highlighted_nodes, ax=ax[0], node_color='blue',
                       node_size=40, label='Optimal Decision')
# ST
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=[229, 443], ax=ax[0], node_color='red', node_size=100, label='source-target nodes')
ax[0].set_title('Baseline(138.566)', fontsize=15, fontweight='normal', y=1.02)

# qr
nx.draw_networkx(G.to_undirected(), position, ax=ax[1], node_color='lightgray', node_size=10, edge_color='lightgray',
                 width=5, alpha=0.3, with_labels=False)
nx.draw_networkx_edges(G.to_undirected(), position, edgelist=x_qr, ax=ax[1], edge_color='blue', width=8)
# 获取高亮路线经过的节点列表
highlighted_nodes = list(set([edge[0] for edge in x_qr] + [edge[1] for edge in x_qr]))
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=highlighted_nodes, ax=ax[1], node_color='blue',
                       node_size=40)
# ST
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=[229, 443], ax=ax[1], node_color='red', node_size=100)
ax[1].set_title('QR(138.566)', fontsize=15, fontweight='normal', y=1.02)

# cqr
nx.draw_networkx(G.to_undirected(), position, ax=ax[2], node_color='lightgray', node_size=10, edge_color='lightgray',
                 width=5, alpha=0.3, with_labels=False)
nx.draw_networkx_edges(G.to_undirected(), position, edgelist=x_cqr, ax=ax[2], edge_color='blue', width=8)
# 获取高亮路线经过的节点列表
highlighted_nodes = list(set([edge[0] for edge in x_cqr] + [edge[1] for edge in x_cqr]))
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=highlighted_nodes, ax=ax[2], node_color='blue',
                       node_size=40)
# ST
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=[229, 443], ax=ax[2], node_color='red', node_size=100)
ax[2].set_title('CQR(132.070)', fontsize=15, fontweight='normal', y=1.02)

# cqr_new
nx.draw_networkx(G.to_undirected(), position, ax=ax[3], node_color='lightgray', node_size=10, edge_color='lightgray',
                 width=5, alpha=0.3, with_labels=False)
nx.draw_networkx_edges(G.to_undirected(), position, edgelist=x_cqr_new, ax=ax[3], edge_color='blue', width=8)
# 获取高亮路线经过的节点列表
highlighted_nodes = list(set([edge[0] for edge in x_cqr_new] + [edge[1] for edge in x_cqr_new]))
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=highlighted_nodes, ax=ax[3], node_color='blue',
                       node_size=40)
# ST
nx.draw_networkx_nodes(G.to_undirected(), position, nodelist=[229, 443], ax=ax[3], node_color='red', node_size=100)
ax[3].set_title('CQR-ERC(132.070)', fontsize=15, fontweight='normal', y=1.02)

# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[3].axis('off')
# plt.subplots_adjust(wspace=1)
legend_labels = {'Optimal Decision (Line)': 'blue', 'Start/End (Node)': 'red'}
legend_handles = [plt.Line2D([], [], color=color, linewidth=8, label=label) for label, color in legend_labels.items()]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize='large')
plt.show()
