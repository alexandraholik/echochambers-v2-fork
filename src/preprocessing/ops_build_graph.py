from src.preprocessing.ops_on_raw_data import check_directory_absence
from src.preprocessing.utilities import (get_only_date, get_metadata,
                                    clean, manage_and_save, add_edge,
                                    create_multi_graph)
#from preprocessing.ops_sentiment_affin import add_sentiment
from src.preprocessing.ops_sentiment_vader import add_sentiment
from src.preprocessing.topic_modelling import add_topic
import networkx as nx
import pandas as pd
from tqdm import tqdm
import os
import ast
import time

def graph_ops():
    garimella_graph()
    # Note: No covid dataset so we can skip this
    # covid_graph()
    vax_graph()
    add_sentiment()
    # add_topic()

# Note: This one takes 'retweet_networks' and makes them into graphs
def garimella_graph():
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/garimella_data')

    # Note: Only runs if /data/garimella_data/Graph does not exist!
    if check_directory_absence('Graph', path):
        os.mkdir('Graph')
        os.chdir(os.path.join(path, 'retweet_networks'))

        # Note: Loads every file in the directory and then proceeds to build graphs for them
        designed_datasets = os.listdir(os.path.join(path, 'retweet_networks'))

        build_garimella_graph(designed_datasets, os.path.join(path, 'Graph'))
    else:
        print("Garilella graph preprocessing skipped...")

    os.chdir(starting_path)

# Note: This one if fast, just creates some .gml files in /Graph
def build_garimella_graph(designed_datasets, path):
    for dataset in designed_datasets:
        df = pd.read_csv(dataset, header=None)
        G_dg = nx.Graph()

        [G_dg.add_edge(row[0], row[1], weight=row[2]) for _, row in df.iterrows()]
        graph_name = dataset.split('_')[2]
        G_dg.name = 'Starter graph' + graph_name
        print(nx.info(G_dg))
        print()

        nx.write_gml(G_dg, path + '/' + graph_name + '.gml')
        G_dg.name = 'Final graph' + graph_name
        print(nx.info(G_dg))
        print()

        G_multi = create_multi_graph(G_dg)
        print(nx.info(G_multi))
        nx.write_gml(G_multi, path + '/Multi_' + graph_name + '.gml')
        print('-----------------------------------------------')

### COVID 19 GRAPHS
def covid_graph():
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/corona_virus')
    meta = False
    if check_directory_absence('Graph', path):
        os.mkdir('Graph')
        os.chdir(os.path.join(path, 'final_data'))
        build_covid_graph(path)
        meta = True
    if meta:
        add_meta(path)
    print('---------------------------------------')
    os.chdir(starting_path)

def build_covid_graph(path):
    df = pd.read_csv(path + '/final_data/' + 'Final_data.csv', usecols = ['username', 'favorites', 'retweets',
                                                                '@mentions', 'geo', 'text_con_hashtag'])
    print(df.columns)
    df.dropna(axis='index', how='all', subset=['text_con_hashtag'], inplace = True)

    G_dg = nx.DiGraph()
    G_g = nx.Graph()

    for _, row in tqdm(df.iterrows(), desc = "Rows processed"):
        if row[4] == 'self' and row[4] != '':
            G_dg = add_edge(G_dg, row[5], 'covid', row[0], row[2], 0, row[3], row[3])
        else:
            try:
                mentions = row[4].split(',')
                for mention in mentions:
                    mention = mention.strip()
                    if mention != '' and mention != '@' :
                        G_dg = add_edge(G_dg, row[5], 'covid', row[0], row[2], 0, row[3], mention)
                        G_g = add_edge(G_g, None, None, None, None, None, row[3], mention)
            except:
                print(row[4])

    G_dg.name = 'Starter covid Direct Graph'
    G_g.name = 'Starter covid Graph'

    graphs = [G_dg, G_g]
    manage_and_save(graphs, path)


def add_meta(path):
    G = nx.read_gml('./Graph/Final_DiGraph_Covid.gml')
    user_metadata_clean = get_metadata()

    not_included = 0
    for node in tqdm(G.nodes(), desc = " Node processed"):
        for column in user_metadata_clean.columns:
            try:
                G.nodes[node][column] = user_metadata_clean.loc[node][column]
            except:
                not_included += 1
                break
    print()
    print("TOTAL NUMBER OF NODES NOT LABELED:{:>10}".format(not_included))
    print("TOTAL NUMBER OF LABELED:          {:>10}".format(len(G) - not_included))
    nx.write_gml(G, path + '/Graph/Final_DiGraph_Covid_data.gml')

### VAX GRAPHS
def vax_graph():
    starting_path = os.getcwd()
    path = os.path.join(starting_path, 'data/vax_no_vax')

    # Note: Same pattern here, if 'Graph' is missing, it will try to generate it
    if check_directory_absence('Graph', path):
        os.mkdir('Graph')
        os.chdir(os.path.join(path, 'final_data'))
        build_vaccination_graph(path)
    else:
        print("Vax graph already built, skipping...")

    os.chdir(starting_path)

# Note: This takes couple of minutes (cca 5min on M3 Mac)
def build_vaccination_graph(path):
    # Reads the data from file that was preprocessed earlier
    df = pd.read_csv(path + '/final_data/' + 'Final_data.csv', lineterminator='\n')

    G_dg = nx.DiGraph()
    G_g = nx.Graph()

    # Note: The processed file
    # 0 - date
    # 1 - username
    # 2 - replies_count
    # 3 - retweets_count
    # 4 - likes_count
    # 5 - hashtags
    # 6 - mentions
    # 7 - tweet
    for _, row in tqdm(df.iterrows(), desc="Rows processed"):
        mentions = ast.literal_eval(row[3])
        if 'self' in mentions:
            G_dg = add_edge(G_dg, row[2], row[7], row[6], row[5], row[4], row[1], row[1])
        else:
            for mention in mentions:
                G_dg = add_edge(G_dg, row[2], row[7], row[6], row[5], row[4], row[1], mention)
                G_g = add_edge(G_g, None, None, None, None, None, row[1], mention)

    G_dg.name = 'Starter vax Direct Graph'
    G_g.name = 'Starter vax Graph'

    graphs = [G_dg, G_g]
    manage_and_save(graphs, path)
