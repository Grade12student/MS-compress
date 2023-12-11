import pickle
import tensorflow as tf
import cv2
import numpy as np
import math
from argparse import ArgumentParser
import imageio

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

def decoder(loadmodel, refer_frame, outputfolder, output_frame):
    graph = load_graph(loadmodel)

    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/train_net_inference_one_pass/train_net/ReconFrame:0')
    res_input = graph.get_tensor_by_name('import/quant_feature:0')
    res_prior_input = graph.get_tensor_by_name('import/quant_z:0')
    motion_input = graph.get_tensor_by_name('import/quant_mv:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')

    with tf.compat.v1.Session(graph=graph) as sess:

        with open(outputfolder + 'quantized_res_feature.pkl', 'rb') as f:
            residual_feature = pickle.load(f)

        with open(outputfolder + 'quantized_res_prior_feature.pkl', 'rb') as f:
            residual_prior_feature = pickle.load(f)

        with open(outputfolder + 'quantized_motion_feature.pkl', 'rb') as f:
            motion_feature = pickle.load(f)

        refer_image = cv2.imread(refer_frame)
        refer_image = refer_image / 255.0
        refer_image = np.expand_dims(refer_image, axis=0)

        # reconstructed image
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

        cv2.imwrite(output_frame, recon_d)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_D.pb', help="decoder model")
    parser.add_argument('--refer_frame', type=str, dest="refer_frame", default='./frame/frame_2_32.png', help="refer frame path")
    parser.add_argument('--loadpath', type=str, dest="outputfolder", default='./testpkl/', help="saved pkl file")
    parser.add_argument('--output_frame', type=str, dest="output_frame", default='./decoded.png', help="output frame path")

    args = parser.parse_args()
    decoder(**vars(args))
