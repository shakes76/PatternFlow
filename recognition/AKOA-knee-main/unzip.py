import argparse
from zipfile import ZipFile

parser = argparse.ArgumentParser()
parser.add_argument("zip_path")
args = parser.parse_args()


def unzip(zip_path, save_dir='data/'):
    '''
    Unzip AKOA data and divide it into left and right ends
    '''
    Z = ZipFile(zip_path)
    # Unzip the data
    for path in Z.namelist():
        # Unzip left knee
        if 'left' in path.lower() or 'l_e_f_t' in path.lower():
            Z.extract(path, save_dir+'left')

    for path in Z.namelist():
        # Unzip right knee
        if 'right' in path.lower() or 'r_i_g_h_t' in path.lower():
            Z.extract(path, save_dir+'right')
    Z.close()


if __name__ == "__main__":
    unzip(args.zip_path, save_dir='data/')
