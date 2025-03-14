import os

import random
from src.controversy_detection.random_walks import random_walks, random_walks_centrality
from src.controversy_detection.change_side_controversy import change_side_controversy
from src.controversy_detection.GMCK import start_GMCK
from src.controversy_detection.EC import start_EC
import logging
from datetime import datetime
import networkx as nx
from src.controversy_detection.log_writer import log_write_start_end
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length


def start_detection():
    logging.basicConfig(filename='controversy_detection.log', level=logging.INFO, format='%(message)s')
    today = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f'RUN TIME: {today}')
    garimella_graph()
    #covid_graph()
    #vax_graph()


def garimella_graph():
    logging.basicConfig(filename='controversy_detection.log', level=logging.INFO, format='%(message)s')
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/garimella_data/Graph')
    os.chdir(path)
    list_of_graph = os.listdir(path)
    log_write_start_end(True, 'GARIMELLA GRAPH')
    for graph in list_of_graph:
        name = graph
        if 'Multi' in name: 
            continue
        else:
            print(name)
            name = name.split('.')[0]
            graph = nx.read_gml(f'{name}.gml')

            logging.info(f'Graph: {name}')
            shortest_path = average_shortest_path_length(graph)
            logging.info(f'Average shortest path: {shortest_path}')

            random_walks(graph, 0.6, shortest_path*2)
            # Note: Needed to add num steps
            random_walks_centrality(graph, num_steps=1)
            print()

            change_side_controversy(graph, 0.6, shortest_path*2)
            print()

            start_GMCK(graph, 'weightComm')
            print()

            #start_EC(graph, 'weightComm')
            #print()
    os.chdir(starting_path)
    log_write_start_end(False)

def covid_graph():
    logging.basicConfig(filename='controversy_detection.log', level=logging.INFO, format='%(message)s')
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/corona_virus/Graph')
    os.chdir(path)
    log_write_start_end(True, 'CORONA VIRUS')
    graph = nx.read_gml('Final_Graph_Covid.gml')
    print(nx.info(graph))

    #shortest_path = average_shortest_path_length(graph, weight='weight')
    shortest_path = 25.596369322723653
    #shortest_path = 25.904721007206568
    #logging.info(f'Average shortest path: {shortest_path}')

    random_walks(graph, 0.6, int(shortest_path*2))
    random_walks(graph, 0.6, int(shortest_path*2), 1)
    random_walks(graph, 0.6, int(shortest_path*2), 2)

    random_walks_centrality(graph)
    random_walks_centrality(graph, 1)
    random_walks_centrality(graph, 2)
    print()

    change_side_controversy(graph, 0.6, shortest_path*2)
    change_side_controversy(graph, 0.6, shortest_path*2, 1)
    change_side_controversy(graph, 0.6, shortest_path*2, 2)
    print()

    start_GMCK(graph, 'weightComm')
    start_GMCK(graph, 'sentimentComm')
    start_GMCK(graph, 'topicComm')
    print()

    #start_EC(graph, 'sentimentComm')
    #print()
    os.chdir(starting_path)
    log_write_start_end(False)

def vax_graph():
    logging.basicConfig(filename='controversy_detection.log', level=logging.INFO, format='%(message)s')
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/vax_no_vax/Graph')
    os.chdir(path)
    log_write_start_end(True, 'Vaccination')
    graph = nx.read_gml('Final_Graph_Vax.gml')

    shortest_path = average_shortest_path_length(graph)
    logging.info(f'Average shortest path: {shortest_path}')

    random_walks(graph, 0.6, shortest_path*2, 1)
    random_walks_centrality(graph, 1)
    print()

    change_side_controversy(graph, 0.6, shortest_path*2, 1)
    print()

    start_GMCK(graph, 'sentimentComm')
    print()

    #start_EC(graph, 'sentimentComm')
    #print()
    os.chdir(starting_path)
    log_write_start_end(False)