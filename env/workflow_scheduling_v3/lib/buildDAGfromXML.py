# assume the runtime in *.xml is the execution time on a CPU with 16 cores
# references: Characterizing and profiling scientific workflows

import xml.etree.ElementTree as ET
# networkx version
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def buildGraph(type, filename, distributed_cloud_enabled, region_map):
    # print("Building DAG:", {filename})
    tot_processTime = 0
    dag = nx.DiGraph(type=type)
    with open(filename, 'rb') as xml_file:
        tree = ET.parse(xml_file)
        xml_file.close()
    root = tree.getroot()
    for child in root:
        # print("Processing job:", {child})
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
            size = 0
            for p in child:
                # print(f"Processing child {p}, of size: {int(p.attrib['size'])}")
                size += int(p.attrib['size'])

            # TODO: Assign region based on dataset mapping, could create some sort of heuristic to extend the dataset by
            #   assigning region ids of tasks with some criteria etc
            # if self.nextTask in self.workflow_dataset_map:
            #     dataset = self.workflow_dataset_map[self.nextTask]
            #     self.processRegion[self.nextTask] = self.dataset_region_map[dataset]
            # else:
            #     # Fallback: Random assignment if no mapping exists (only for testing)
            random_region = np.random.choice(3) if distributed_cloud_enabled else 0
            dag.add_node(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) * 16, size=size,
                         regionId=random_region)
            tot_processTime += float(child.attrib['runtime']) * 16

            # print(f"Total Process time of job {int(child.attrib['id'][2:])}: {float(child.attrib['runtime']) * 16}")
            # print(f"Total Size of job {int(child.attrib['id'][2:])}: {size}")

            # dag.add_node(child.attrib['id'], processTime=float(child.attrib['runtime'])*16, size=size)
        if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
            kid = int(child.attrib['ref'][2:])
            for p in child:
                parent = int(p.attrib['ref'][2:])
                dag.add_edge(parent, kid)

    # print(f"Total Process time for {filename}: {tot_processTime}")

    # Toggle to draw DAG's of each representation built as part of a list of workflows.
    # draw_dag(dag, region_map, f"{filename}.png")

    return dag, tot_processTime


def draw_dag(dag, region_map, save_path=None):
    from networkx.drawing.nx_agraph import graphviz_layout

    # Generate a hierarchical layout (top-down)
    pos = graphviz_layout(dag, prog='dot')
    plt.figure(figsize=(12, 8))

    # Assign unique colors to each region ID
    region_ids = {dag.nodes[node].get('regionId') for node in dag.nodes()}
    color_map = {region: color for region, color in zip(region_ids, mcolors.TABLEAU_COLORS)}

    # Get colors for nodes based on their region ID
    node_colors = [color_map[dag.nodes[node].get('regionId')] for node in dag.nodes()]

    # Create labels with node number and process time
    node_labels = {
        node: f"{node}\n{dag.nodes[node]['processTime']}s"
        for node in dag.nodes()
    }

    nx.draw(
        dag,
        pos,
        with_labels=False,
        node_size=1000,
        node_color=node_colors,  # Use region-based colors
        edge_color='gray',
        font_size=8
    )

    nx.draw_networkx_labels(
        dag,
        pos,
        labels=node_labels,
        font_size=8,
        font_color='black'
    )

    # Create a legend for region IDs
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color=color,
            markersize=10,
            linestyle='',
        ) for color in color_map.values()
    ]
    plt.legend(
        handles,
        [region_map[color] for color in color_map.keys()],
        title="Region ID",
        loc="lower right",
        fontsize='small',
        title_fontsize='medium'
    )

    # Draw edges
    nx.draw_networkx_edges(dag, pos, edgelist=dag.edges(), edge_color='gray')

    plt.title("Distributed_CyberShake_30")  # TODO: Add dynamic name for title

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")

    plt.show()

# # ============ testing =============
# options = {
#     'node_color': 'black',
#     'node_size': 10,
#     'width': 2,
#     'arrowstyle': '-|>',
#     'arrowsize': 10,
# }
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, parentdir)
# test = buildGraph('CyberShake', parentdir+'/dax/CyberShake_30.xml')
# nx.draw_networkx(test, arrows=True, **options)
# nx.draw(test)
# plt.show()



# import igraph
# import matplotlib.pyplot as plt
# import xml.etree.ElementTree as ET
#
#
# def buildGraph(type, filename):
#     dag = igraph.Graph(directed=True)
#     dag["type"] = type
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     for child in root:
#         if child.tag == '{http://pegasus.isi.edu/schema/DAX}job':
#             size = 0
#             for p in child:
#                 size += int(p.attrib['size'])
#             dag.add_vertex(int(child.attrib['id'][2:]), processTime=float(child.attrib['runtime']) * 16, size=size)
#             # dag.add_node(child.attrib['id'], processTime=float(child.attrib['runtime'])*16, size=size)
#         if child.tag == '{http://pegasus.isi.edu/schema/DAX}child':
#             kid = int(child.attrib['ref'][2:])
#             for p in child:
#                 parent = int(p.attrib['ref'][2:])
#                 dag.add_edge(parent, kid)
#     return dag
#
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, parentdir)
# test = buildGraph('CyberShake', parentdir+'/dax/CyberShake_30.xml')
# layout = test.layout("kk")
# print(test.vs.indices)
# print(f'Expected output: 3.04, Actual output: {test.vs[1]["processTime"]}')
# # igraph.plot(test, layout=layout)  # not working because require Cairo library https://stackoverflow.com/questions/12072093/python-igraph-plotting-not-available/45416251#45416251

