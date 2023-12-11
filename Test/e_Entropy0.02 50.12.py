import tensorflow as tf
import imageio
import numpy as np
import os
import pickle
import math
from argparse import ArgumentParser
import heapq
from collections import defaultdict


# Implement Huffman coding for entropy encoding
def build_huffman_tree(freq_dict):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def encode_with_huffman(data, huffman_tree):
    encoded_data = ""
    codebook = {symbol: code for symbol, code in huffman_tree}
    for symbol in data:
        encoded_data += codebook[symbol]
    return encoded_data

def load_graph(frozen_graph_filename):
    with open(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def encoder(loadmodel, input_path, refer_path, outputfolder):
    graph = load_graph(loadmodel)
    prefix = 'import/build_towers/tower_0/train_net_inference_one_pass/train_net/'

    Res = graph.get_tensor_by_name(prefix + 'Residual_Feature:0')
    inputImage = graph.get_tensor_by_name('import/input_image:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')
    Res_prior = graph.get_tensor_by_name(prefix + 'Residual_Prior_Feature:0')
    motion = graph.get_tensor_by_name(prefix + 'Motion_Feature:0')
    bpp = graph.get_tensor_by_name(prefix + 'rate/Estimated_Bpp:0')
    psnr = graph.get_tensor_by_name(prefix + 'distortion/PSNR:0')
    # reconstructed frame
    reconframe = graph.get_tensor_by_name(prefix + 'ReconFrame:0')

    im1 = imageio.imread(input_path)
    im2 = imageio.imread(refer_path)
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im1 = np.expand_dims(im1, axis=0)
    im2 = np.expand_dims(im2, axis=0)

    bpp_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = tf.compat.v1.Session(graph=graph).run(
        [bpp, Res, Res_prior, motion, psnr, reconframe], feed_dict={
            inputImage: im1,
            previousImage: im2
        })

    print(bpp_est)
    print(psnr_val)
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    # Using Huffman coding for encoding
    flattened_res_q = Res_q.flatten()
    freq_dict = defaultdict(int)
    for symbol in flattened_res_q:
        freq_dict[symbol] += 1

    huffman_tree = build_huffman_tree(freq_dict)
    encoded_data = encode_with_huffman(flattened_res_q, huffman_tree)

    with open(outputfolder + 'encoded_res_data.txt', 'w') as f:
        f.write(encoded_data)

    with open(outputfolder + 'huffman_tree.pkl', 'wb') as output:
        pickle.dump(huffman_tree, output)

    with open(outputfolder + 'quantized_res_prior_feature.pkl', 'wb') as output:
        pickle.dump(Res_prior_q, output)

    with open(outputfolder + 'quantized_motion_feature.pkl', 'wb') as output:
        pickle.dump(motion_q, output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb', help="encoder model")
    parser.add_argument('--input_frame', type=str, dest="input_path", default='./frame', help="input frame folder")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./frame', help="refer frame folder")
    parser.add_argument('--outputpath', type=str, dest="outputfolder", default='./testpkl/', help="output pkl folder")

    args = parser.parse_args()

    input_folder = args.input_path
    refer_folder = args.refer_path

    input_images = sorted(os.listdir(input_folder))
    refer_images = sorted(os.listdir(refer_folder))

    for input_image, refer_image in zip(input_images, refer_images):
        input_image_path = os.path.join(input_folder, input_image)
        refer_image_path = os.path.join(refer_folder, refer_image)

        encoder(args.loadmodel, input_image_path, refer_image_path, args.outputfolder)