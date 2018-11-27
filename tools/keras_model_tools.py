# Python script of tools for operations on Keras models

import argparse
import h5py


class KModelTools:
    def __init__(self, h5_path=None):
        self.h5_path = h5_path
        self.f_h5 = h5py.File(h5_path)

    def print_h5_wegiths(self):
        for layer, g in self.f_h5.items():
            print("{}".format(layer))
            for key, value in g.attrs.items():
                print("{}: {}".format(key, value)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for Keras model')
    parser.add_argument('--h5', type=str, required=True, help='Path to the Keras model')
    args = parser.parse_args()

    k_model_tools = KModelTools(args.h5)
    k_model_tools.print_h5_wegiths()
