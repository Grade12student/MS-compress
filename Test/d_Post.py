import tensorflow as tf
import imageio
import numpy as np
import os
import pickle
import math
from argparse import ArgumentParser
import cv2  # Import OpenCV for image processing

def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def load_graph(frozen_graph_filename):
    with open(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def post_process(image):
    # Convert the float32 image to uint8 format
    image_uint8 = (255.0 * np.clip(image, 0, 1)).astype(np.uint8)

    # Apply deblocking filter to the image
    filtered_image = cv2.fastNlMeansDenoisingColored(image_uint8, None, 10, 10, 7, 21)

    return filtered_image

def decoder(loadmodel, refer_path, outputfolder):
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

        im1 = imageio.imread(refer_path)
        im1 = im1 / 255.0
        im1 = np.expand_dims(im1, axis=0)

        # reconstructed image
        recon_d = sess.run(
        [reconframe],
        feed_dict={
            res_input: residual_feature,
            res_prior_input: residual_prior_feature,
            motion_input: motion_feature,
            previousImage: im1
        })

        # Apply post-processing deblocking filter
        recon_d_processed = post_process(recon_d[0][0])

        filename = os.path.splitext(os.path.basename(refer_path))[0]
        imageio.imwrite(os.path.join(outputfolder, filename + '_reconstructed_image.png'), recon_d_processed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb', help="decoder model")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./frame', help="refer frame folder")
    parser.add_argument('--loadpath', type=str, dest="outputfolder", default='./testpkl/', help="saved pkl file")
   
    args = parser.parse_args()

    refer_folder = args.refer_path
    output_folder = args.outputfolder

    refer_images = sorted(os.listdir(refer_folder))

    for refer_image in refer_images:
        refer_image_path = os.path.join(refer_folder, refer_image)
        decoder(args.loadmodel, refer_image_path, output_folder)