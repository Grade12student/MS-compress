import pickle
import tensorflow as tf
import cv2
import numpy as np
import math
from argparse import ArgumentParser
import os

def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def)
    return graph

def resize_frames(frames, target_shape):
    resized_frames = [cv2.resize(frame, target_shape, interpolation=cv2.INTER_AREA) for frame in frames]
    return resized_frames

def decoder(loadmodel, refer_frame, outputfolder, output_frame):
    graph = load_graph(loadmodel)
    sess = tf.compat.v1.Session(graph=graph)  # Initialize the TensorFlow session

    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/train_net_inference_one_pass/train_net/ReconFrame:0')
    res_input = graph.get_tensor_by_name('import/quant_feature:0')
    res_prior_input = graph.get_tensor_by_name('import/quant_z:0')
    motion_input = graph.get_tensor_by_name('import/quant_mv:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')

    residual_feature = None  # Initialize the variable to avoid any potential issues

    if os.path.isfile(refer_frame):
        # If the refer_frame argument is a file, process the single frame
        refer_image = cv2.imread(refer_frame)
        if refer_image is None:
            raise ValueError(f"Failed to load the reference image from {refer_frame}")
        refer_image = refer_image / 255.0
        refer_image = np.expand_dims(refer_image, axis=0)
        output_folder = output_frame

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # ... (rest of the code remains the same)
    elif os.path.isdir(refer_frame):
        # If the refer_frame argument is a directory, process all frames in the directory
        i = 0  # Initialize a counter for incrementing filenames
        for filename in os.listdir(refer_frame):
            refer_image = cv2.imread(os.path.join(refer_frame, filename))
            if refer_image is None:
                print(f"Failed to load the reference image from {os.path.join(refer_frame, filename)}")
                continue
            refer_image = refer_image / 255.0
            refer_image = np.expand_dims(refer_image, axis=0)
            output_folder = output_frame

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            with open(outputfolder + 'quantized_res_feature.pkl', 'rb') as f:
                residual_feature = pickle.load(f)
                # Ensure the shape matches the graph's expected shape
                residual_feature = np.expand_dims(residual_feature, axis=0)  # Add this line to adjust the dimensions
                residual_feature = np.squeeze(residual_feature, axis=0)  # Adjust the shape accordingly
                #print("Residual Feature Shape:", residual_feature.shape)  # Add this line to print the shape of the data

            with open(outputfolder + 'quantized_res_prior_feature.pkl', 'rb') as f:
                residual_prior_feature = pickle.load(f)
                # Ensure the shape matches the graph's expected shape
                residual_prior_feature = np.expand_dims(residual_prior_feature, axis=0)  # Add this line to adjust the dimensions
                residual_prior_feature = np.squeeze(residual_prior_feature, axis=0)  # Adjust the shape accordingly
                #print("Residual Prior Feature Shape:", residual_prior_feature.shape)  # Add this line to print the shape of the data

            with open(outputfolder + 'quantized_motion_feature.pkl', 'rb') as f:
                motion_feature = pickle.load(f)
                # Ensure the shape matches the graph's expected shape
                motion_feature = np.expand_dims(motion_feature, axis=0)  # Add this line to adjust the dimensions
                motion_feature = np.squeeze(motion_feature, axis=0)  # Adjust the shape accordingly
                #print("Motion Feature Shape:", motion_feature.shape)  # Add this line to print the shape of the data

            if residual_feature is not None:  # Check if residual_feature has been initialized
                # Reconstructed image
                recon_d = sess.run(
                    [reconframe],
                    feed_dict={
                        res_input: residual_feature,
                        res_prior_input: residual_prior_feature,
                        motion_input: motion_feature,
                        previousImage: refer_image
                    })

                # Convert the output to a valid format
                recon_d = np.squeeze(recon_d[0])  # Assuming recon_d is a NumPy array
                recon_d = np.clip(recon_d, 0, 1)  # Clip values to ensure they are in the valid range
                recon_d = (recon_d * 255).astype(np.uint8)  # Convert the values to the appropriate range for saving as an image

                # Save the output with an incremented filename
                cv2.imwrite(os.path.join(output_folder, f"decoded_{i}.png"), recon_d)

                i += 1  # Increment the counter for the next filename

    else:
        raise ValueError(f"Invalid input provided for refer_frame: {refer_frame}")

    sess.close()  # Close the TensorFlow session

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_D.pb', help="decoder model")
    parser.add_argument('--refer_frame', type=str, dest="refer_frame", default='./frame', help="refer frame path")
    parser.add_argument('--loadpath', type=str, dest="outputfolder", default='./E/', help="saved pkl file")
    parser.add_argument('--output_frame', type=str, dest="output_frame", default='./D/', help="output frame directory")

    args = parser.parse_args()
    decoder(**vars(args))
