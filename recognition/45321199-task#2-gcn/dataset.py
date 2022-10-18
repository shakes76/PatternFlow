import numpy as np
from sklearn.preprocessing import LabelBinarizer as lb
from tensorflow.keras import utils


class DataLoader:
    def download_data(self):
        data_url = "https://graphmining.ai/datasets/ptg/facebook.npz"
        data_fname = "facebook.npz"

        self.data_dir = utils.get_file(data_fname, data_url)
        print(f"Data saved at {self.data_dir}")

    def process_data(self):
        # Unpack data into components
        with np.load(self.data_dir) as data:
            edges, features, labels = data.values()

        len_label_types = len(np.unique(labels))
        len_vertices = features.shape[0]
        len_features = features.shape[1]
        len_edges = len(edges) // 2

        # Print dimensions of data
        print(
            f"Label Types:\t{len_label_types}",
            f"Edges:      \t{len_edges}",
            f"Vertices:   \t{len_vertices}",
            f"Features:   \t{len_features}",
            sep='\n'
        )

        # get blank adjacency matrix representation of graph
        graph_adj_mat = self.adj_mat_repr(len_vertices)

        # one-hot encode labels i.e. 0,1,2,3 -> 1000,0100,0010,0001
        encoder = lb()
        encoded_labels = encoder.fit_transform(labels)

        #retrieve range for each subset of the data
        train_range, val_range, test_range = self.split_data(len_vertices)

        # split data into train/validation/test; 0.2:0.2:0.6 ratio
        train_mask = np.zeros(len_vertices, dtype=np.bool)
        val_mask = np.zeros(len_vertices, dtype=np.bool)
        test_mask = np.zeros(len_vertices, dtype=np.bool)

        # Label each vertice with training, validation or test
        train_mask[train_range] = True
        val_mask[val_range] = True
        test_mask[test_range] = True

        # package processed data
        self.processed_data = {
            "train_mask" : train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "graph_adj_mat" : graph_adj_mat,
            "encoded_labels" : encoded_labels,
            "features" : features,
            "len_label_types" : len_label_types,
            "len_vertices" : len_vertices ,
            "len_features" : len_features ,
            "len_edges" : len_edges
        }

    def split_data(self, len_vertices):
        train_range = range(len_vertices // 5)
        val_range = range(len_vertices // 5, 2 * len_vertices//5)
        test_range = range(2 * len_vertices // 5, len_vertices)
        return train_range, val_range, test_range
    
    @staticmethod
    def adj_mat_repr(len_vertices):
        return np.eye(len_vertices, dtype=np.float64)

    def load_data(self):
        self.download_data()
        self.process_data()
        return self.processed_data
            