# Python script of tools for operations on TensorFlow models

import argparse
import tensorflow as tf
import cv2
from tensorflow.python import pywrap_tensorflow


class ModelTools:
    def __init__(self, ckpt_path=None, pb_path=None):
        self.ckpt_path = ckpt_path
        self.pb_path = pb_path
        if pb_path is not None:
            # Load the model of the .pb format
            graph_def = tf.GraphDef.FromString(open(self.pb_path, 'rb').read())
            tf.Graph().as_default()
            tf.import_graph_def(graph_def, name='')

    def print_ckpt_tensor(self):
        # Read data from checkpoint file
        reader = pywrap_tensorflow.NewCheckpointReader(self.ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # Print tensor name and shape
        for key in var_to_shape_map:
            print("tensor: ", key, reader.get_tensor(key).shape,
                  reader.get_tensor(key).dtype)  # print the name and the shape
            # print(reader.get_tensor(key))  # print the value

    def print_pb_tensor(self): 
        ops = tf.get_default_graph().get_operations()
        for op in ops:
            print(op.name)
            print(op.values())
        # print(tf.get_default_graph().get_tensor_by_name("input_image:0"))

    # list to announce the input and output tensors
    def test_inference(self, img, input_tensors, output_tensors):
        img_raw = cv2.imread(img)
        session = tf.Session()
        feed_dict = {input_tensors[0]: img_raw}
        frame_features = session.run(output_tensors, feed_dict=feed_dict)
        print(frame_features[0].shape)  # Debug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tool set for TensorFlow model')
    parser.add_argument('--ckpt', type=str, required=False, help='Path to the TensorFlow checkpoint')
    parser.add_argument('--pb', type=str, required=False, help='Path to the TensorFlow frozen graph')
    parser.add_argument('--img', type=str, required=False, help='Path to an image for test')
    args = parser.parse_args()

    model_tools = ModelTools(ckpt_path=args.ckpt, pb_path=args.pb)
    model_tools.print_pb_tensor()
