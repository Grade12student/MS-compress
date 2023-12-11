import pickle
import tensorflow as tf
import imageio.v2 as imageio
import numpy as np
from argparse import ArgumentParser
import cv2  # For image processing
import os

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

# Generate low-resolution versions of frames
def generate_low_resolution_frames(frames):
    # Use downsampling or wavelet transform to create low-resolution versions
    low_res_frames = [cv2.resize(frame, (208, 112)) for frame in frames]
    low_res_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in low_res_frames]
    return low_res_frames

# Save the low-resolution frames
def save_low_resolution_frames(low_res_frames, outputfolder):
    for i, frame in enumerate(low_res_frames):
        # Ensure that the color mode is preserved properly during saving
        if frame.shape[2] == 1: # grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) # Convert to RGB
        imageio.imwrite(os.path.join(outputfolder, f"low_res_frame_{i}.png"), frame)

# Implement variable density sampling strategy using DFT
def variable_density_sampling(frames):
    for i, frame in enumerate(frames):
        dft_components = []
        for channel in range(frame.shape[2]):
            # Apply 2D Discrete Fourier Transform (DFT) on each channel
            dft = cv2.dft(np.float32(frame[:, :, channel]), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

            # Create a mask for variable density sampling
            mask = np.zeros_like(frame[:, :, channel])

            # Set higher sampling density for high-frequency components
            # Adjust the threshold as needed based on the frequency spectrum
            threshold = 100
            mask[magnitude_spectrum > threshold] = 1

            # Apply the mask to the frequency domain
            dft_shift[:, :, 0] *= mask
            dft_shift[:, :, 1] *= mask

            # Shift back the zero-frequency component and apply inverse DFT
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            # Append the channel component to the list
            dft_components.append(img_back)

        # Combine the DFT results from all channels
        img_back = np.stack(dft_components, axis=-1)

        # Optional: Rescale or normalize the reconstructed frame
        img_back = np.uint8(img_back / img_back.max() * 255)

        # Save the reconstructed frame
        imageio.imwrite(os.path.join('./testpkl/', f"reconstructed_frame_{i}.png"), img_back)


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

    with tf.compat.v1.Session(graph=graph) as sess:
        im1 = imageio.imread(input_path)
        im2 = imageio.imread(refer_path)
        im1 = im1 / 255.0
        im2 = im2 / 255.0
        im1 = np.expand_dims(im1, axis=0)
        im2 = np.expand_dims(im2, axis=0)

        bpp_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = sess.run(
            [bpp, Res, Res_prior, motion, psnr, reconframe], feed_dict={
                inputImage: im1,
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


# Use the new functions in the encoder
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb', help="encoder model")
    parser.add_argument('--frame_folder', type=str, dest="frame_folder", default='./frame/', help="folder containing frames")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./frame/frame_2_32.png', help="refer frame path")
    parser.add_argument('--output_folder', type=str, dest="outputfolder", default='./testpkl/', help="output folder")

    args = parser.parse_args()

    frame_paths = [os.path.join(args.frame_folder, frame) for frame in os.listdir(args.frame_folder)]
    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    low_res_frames = generate_low_resolution_frames(frames)
    save_low_resolution_frames(low_res_frames, args.outputfolder)
    variable_density_sampling(frames)  # Implement your sampling strategy here
    encoder(loadmodel=args.loadmodel, input_path=frame_paths[0], refer_path=args.refer_path, outputfolder=args.outputfolder)