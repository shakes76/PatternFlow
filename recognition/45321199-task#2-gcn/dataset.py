import wget
import os

class DataLoader:
    def __init__(self):
        pass
    
    def download_data(self):
        """
            Retrieves the facebook.npz dataset if it doesn't already exist.
        """
        data_url = "https://graphmining.ai/datasets/ptg/facebook.npz"
        data_path = "facebook.npz"

        if not os.path.exists(data_path):
            print(f'Downloading from {data_url}, this may take a while...')
            self.filename = wget.download(data_url)
        else:
            print("Data has already been retrieved")
    
            