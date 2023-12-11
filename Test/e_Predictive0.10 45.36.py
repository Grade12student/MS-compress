import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, UpSampling2D, Concatenate
import cv2
import imageio
import numpy as np
import os
import pickle
import math
from argparse import ArgumentParser


class PredictiveCoder:

    def __init__(self):
        # Motion compensation model
        self.motion_model = self.build_motion_model()

        # ConvLSTM for multi-frame prediction
        self.convlstm = ConvLSTM2D(filters=16, kernel_size=(3, 3))

        # Multiple prediction modes
        # Intra prediction
        self.intra_pred = tf.keras.Sequential([
            Conv2D(8, (3, 3), padding="same"),
            UpSampling2D((2, 2))
        ])

        # Inter prediction
        self.inter_pred = tf.keras.Sequential([
            Concatenate(),
            Conv2D(8, (3, 3), padding="same"),
            UpSampling2D((2, 2))
        ])

        # Bi-directional prediction
        self.bi_pred = tf.keras.Sequential([
            Concatenate(),
            ConvLSTM2D(8, (3, 3)),
            Conv2D(8, (3, 3), padding="same"),
            UpSampling2D((2, 2))
        ])

    def build_motion_model(self):
        conv_layer = Conv2D(filters=32, kernel_size=(3, 3), padding="same")

        @tf.function
        def motion_model(inputs):
            x = conv_layer(inputs)
            return x

        return motion_model


    def predict(self, input_frames):

        frame1, frame2 = input_frames
        
        flow1 = self.motion_model(tf.concat([frame1, frame2], axis=-1))
        
        flow2 = self.motion_model(frame2)
        
        aligned1 = cv2.remap(frame1, flow1)
        aligned2 = cv2.remap(frame2, flow2)

        predicted = self.convlstm([aligned1, aligned2])
        # Motion compensation
        flows = self.motion_model(input_frames)
        aligned_frames = [cv2.remap(input_frames[i], flows[i]) for i in range(len(input_frames))]

        # Multi-frame prediction
        predicted_frame = self.convlstm(aligned_frames)

        # Different modes
        intra_pred = self.intra_pred(input_frames[-1])
        inter_pred = self.inter_pred(aligned_frames[-2], aligned_frames[-1])
        bi_pred = self.bi_pred(aligned_frames[-2], aligned_frames[-1], flows[-1])

        # Compute costs
        intra_cost = tf.reduce_mean(tf.square(intra_pred - input_frames[-1]))
        inter_cost = tf.reduce_mean(tf.square(inter_pred - input_frames[-1]))
        bi_cost = tf.reduce_mean(tf.square(bi_pred - input_frames[-1]))

        # Take best prediction
        if intra_cost < inter_cost and intra_cost < bi_cost:
            pred = intra_pred
        elif inter_cost < intra_cost and inter_cost < bi_cost:
            pred = inter_pred
        else:
            pred = bi_pred

        return pred


def load_graph(frozen_graph_filename):
    with open(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph


def encoder(loadmodel, input_path, refer_path, outputfolder):
    coder = PredictiveCoder()
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

    with tf.compat.v1.Session(graph=graph) as sess:
        im1 = imageio.imread(input_path)
        im2 = imageio.imread(refer_path)
        im1 = im1 / 255.0
        im2 = im2 / 255.0
        im1 = np.expand_dims(im1, axis=0)
        im2 = np.expand_dims(im2, axis=0)

        # Apply predictive coding
        predicted_input = im1 - im2  # Simple differencing as a predictive coding method
        predicted_input = np.clip(predicted_input, 0, 1)  # Ensure values are within [0, 1] range

        coder.predict([im1, im2])  # Use the predicted input for encoding

        bpp_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = sess.run(
            [bpp, Res, Res_prior, motion, psnr, reconframe], feed_dict={
                inputImage: predicted_input,
                previousImage: im2
            })

        print(bpp_est)
        print(psnr_val)
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        with open(outputfolder + 'quantized_res_feature.pkl', 'wb') as output:
            pickle.dump(Res_q, output)

        with open(outputfolder + 'quantized_res_prior_feature.pkl', 'wb') as output:
            pickle.dump(Res_prior_q, output)

        with open(outputfolder + 'quantized_motion_feature.pkl', 'wb') as output:
            pickle.dump(motion_q, output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb',
                        help="encoder model")
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
