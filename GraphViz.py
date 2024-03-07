import networkx as nx
from Preprocessing import Preprocessing
import numpy as np
import matplotlib.pyplot as plt

class GraphViz():
    def __init__(self,label,list_event,list_mention):
        self.label = label
        self.list_event = list_event
        self.list_mention = list_mention
    
    def _retrieve_data_for_viz(self):
        preprocessing = Preprocessing(self.label)
        labels,df_events,df_mentions = preprocessing.data_load(self.list_event,self.list_mention)
        return preprocessing.create_graph(labels,df_events,df_mentions,mode="analyse")
    
    def display_graph(self):
        
        df_article_sorted,labels_sorted,df_events_sorted_temp,edge_mentionné,edge_est_source_de,y = self._retrieve_data_for_viz()
        
        G = nx.MultiDiGraph()

        ag_list = []
        sgt_list = []
        sgf_list = []
        sgn_list = []
        eg_list = []
        for i, row in df_article_sorted.iterrows():
            G.add_node(f"{i}_a", node_type='article', x=row.to_numpy().astype(np.float32))
            if self.label == "article":
                if y[i] == 1:
                    sgf_list.append(f"{i}_a")
                if y[i] == 0:
                    sgt_list.append(f"{i}_a")
                else:
                    sgn_list.append(f"{i}_a")         
            else: 
                ag_list.append(f"{i}_a")
                
        for i, row in labels_sorted.iterrows():
            G.add_node(f"{i}_s", node_type='source', x=row.to_numpy().astype(np.float32))
            if self.label == "source":
                if y[i] == 1:
                    sgf_list.append(f"{i}_s")
                if y[i] == 0:
                    sgt_list.append(f"{i}_s")
                else:
                    sgn_list.append(f"{i}_s")
            else: 
                ag_list.append(f"{i}_s")
        for i, row in df_events_sorted_temp.iterrows():
            G.add_node(f"{i}_e", node_type='event', x=row.to_numpy().astype(np.float32))
            eg_list.append(f"{i}_e")

        def append_suffix_mentionne(element, row_index):
            if row_index == 0:
                return str(element) + '_e'
            elif row_index == 1:
                return str(element) + '_a'
        
        def append_suffix_edge_est_source_de(element, row_index):
            if row_index == 0:
                return str(element) + '_s'
            elif row_index == 1:
                return str(element) + '_a'
        
        edge_mentionné_appended = np.vectorize(append_suffix_mentionne)(edge_mentionné, np.indices(edge_mentionné.shape)[0])
        edge_est_source_de_appended = np.vectorize(append_suffix_edge_est_source_de)(edge_est_source_de, np.indices(edge_est_source_de.shape)[0])


        G.add_edges_from(edge_mentionné_appended.transpose())
        G.add_edges_from(edge_est_source_de_appended.transpose())

        isolated_nodes = [node for node, degree in G.degree() if degree == 0]
        G.remove_nodes_from(isolated_nodes)

        G = G.to_undirected()

        pos = nx.spring_layout(G)

        plt.figure(figsize=(12,12))

        nx.draw_networkx_edges(G, pos)

        nx.draw_networkx_nodes(G,
                pos,
                nodelist = [x for x in ag_list if x in G.nodes],
                node_size = 20,
                node_color = "y",
                node_shape = 'o',
                alpha=0.9,
            )

        nx.draw_networkx_nodes(G,
                pos,
                nodelist = [x for x in sgf_list if x in G.nodes],
                node_size = 200,
                node_color = "r",
                node_shape = 'o',
                alpha=0.9,
            )

        nx.draw_networkx_nodes(G,
                pos,
                nodelist = [x for x in sgt_list if x in G.nodes],
                node_size = 20,
                node_color = "g",
                node_shape = 'o',
                alpha=0.9,
            )

        nx.draw_networkx_nodes(G,
                pos,
                nodelist = [x for x in sgn_list if x in G.nodes],
                node_size = 20,
                node_color = "b",
                node_shape = 'o',
                alpha=0.9,
            )

        nx.draw_networkx_nodes(G,
                pos,
                nodelist = [x for x in eg_list if x in G.nodes],
                node_size = 20,
                node_color = 'm',
                node_shape = 's',
                alpha=0.9,
            )
        plt.show()
        
if __name__ == "__main__":
    
    label = "article"
    list_mention = ["gdelt_data/20231001000000.mentions.CSV","gdelt_data/20231001001500.mentions.CSV"]
    list_event = ["gdelt_data_event/20231001000000.export.CSV","gdelt_data_event/20231001001500.export.CSV"]
    
    analyse = GraphViz(label,list_event,list_mention)
    analyse.display_graph()