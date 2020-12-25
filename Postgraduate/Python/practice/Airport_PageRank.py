import networkx as nx
import matplotlib.pyplot as plt


flights_network = nx.read_edgelist('./input/out.opsahl-openflights.csv', create_using=nx.DiGraph())

print('航班数：' + str(len(flights_network.nodes)))
print('航线数：' + str(len(flights_network.edges)))

fig, ax = plt.subplots(figsize=(24, 16))
pos_flights = nx.kamada_kawai_layout(flights_network)
ax.axis('off')
plt.box(False)
nx.draw(flights_network, node_size=30, node_color='green', edge_color='#D8D8D8', width=0.3, ax=ax)
plt.show()

















