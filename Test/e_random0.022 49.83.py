import tensorflow as tf
import imageio
import numpy as np
import os
import pickle
from argparse import ArgumentParser
from skimage.measure import block_reduce
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Lasso
from scipy.fftpack import dct, idct
from pywt import dwt2

# Add sparsity-inducing transform function
def apply_transform(image, transform_type='wavelet'):
    if transform_type == 'dct':
        return dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    elif transform_type == 'wavelet':
        # Apply wavelet transform
        coeffs = dwt2(image, 'bior1.3')
        return coeffs[0]  # LL subband approximation
    else:
        raise ValueError("Invalid transform type")

# Add code for matrix optimization (dictionary learning or matrix factorization)
def optimize_measurement_matrix(image, num_measurements):
    # Flatten the image
    flattened_image = image.flatten()

    # Use MiniBatchDictionaryLearning to learn the measurement matrix
    dico = MiniBatchDictionaryLearning(n_components=num_measurements, alpha=1, n_iter=5)
    dico.fit(flattened_image.reshape(1, -1))

    # Return the learned measurement matrix
    return dico.components_

# Add advanced quantization scheme (e.g., Sigma-Delta)
def sigma_delta_quantization(data):
    # Perform Sigma-Delta quantization
    quantized_data = np.zeros_like(data)
    for i in range(1, len(data)):
        quantized_data[i] = quantized_data[i - 1] + np.round(data[i] - quantized_data[i - 1])

    return quantized_data.astype(int)

# Add denoising filter function
def denoise(image):
    # Implement your denoising filter here
    # Example: You can use a Gaussian filter or other denoising techniques
    return image

# Add reconstruction algorithm (e.g., LASSO)
def lasso_reconstruction(encoded_measurements, measurement_matrix):
    # Use LASSO for reconstruction
    lasso = Lasso(alpha=0.01)
    lasso.fit(measurement_matrix, encoded_measurements)
    reconstructed_image = lasso.coef_.reshape((measurement_matrix.shape[1], 1))

    return reconstructed_image

# Add function to estimate bitrate and distortion
def estimate_bitrate_and_distortion(reconstructed_image, original_image, lambda_factor):
    # Assuming original_image and reconstructed_image are 2D arrays
    bitrate_est = len(reconstructed_image.flatten()) * lambda_factor  # Adjust lambda for desired tradeoff
    distortion_est = np.mean((reconstructed_image - original_image) ** 2)

    return bitrate_est, distortion_est

# Add compressed sensing functions
def random_measurements(image, measurement_matrix):
    measurements = np.dot(measurement_matrix, image.flatten())
    return measurements

def iterative_reconstruction(encoded_measurements, measurement_matrix, initial_guess, num_iterations):
    current_guess = initial_guess.copy()
    for _ in range(num_iterations):
        decoded_measurements = np.dot(measurement_matrix.T, current_guess.flatten())
        error = encoded_measurements - decoded_measurements
        current_guess += np.dot(measurement_matrix, error.reshape(current_guess.shape))
    return current_guess.reshape(initial_guess.shape)

# Quantization Function
def quantize_function(data, bit_depth=8):
    quantized_data = np.round(data * (2**bit_depth - 1)) / (2**bit_depth - 1)
    return quantized_data

def load_graph(frozen_graph_filename):
    with open(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
    return graph

def encoder(loadmodel, input_path, refer_path, outputfolder, num_measurements, bit_depth=8, lambda_factor=0.1):
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

    # Modify the input image shape to match the expected shape by adding a batch dimension
    im1 = imageio.imread(input_path)
    im2 = imageio.imread(refer_path)
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im1 = np.expand_dims(im1, axis=0)  # Add batch dimension
    im2 = np.expand_dims(im2, axis=0)  # Add batch dimension
    im1_transformed = apply_transform(im1.squeeze())

    # Optimize the measurement matrix
    measurement_matrix = optimize_measurement_matrix(im1_transformed, num_measurements)

    # Take random measurements in the transform domain
    encoded_measurements = random_measurements(im1_transformed.flatten(), measurement_matrix)

    # Quantize and encode the random measurements
    quantized_measurements = quantize_function(encoded_measurements, bit_depth)
    entropy_encoded_data = sigma_delta_quantization(quantized_measurements)

    # Estimate bitrate and distortion
    bitrate_est, Res_q, Res_prior_q, motion_q, psnr_val, recon_val = tf.compat.v1.Session(graph=graph).run(
        [bpp, Res, Res_prior, motion, psnr, reconframe], feed_dict={
            inputImage: im1,
            previousImage: im2
        })

    print("Original BPP:", bitrate_est)
    print("Original PSNR:", psnr_val)

    # Denoise the reconstructed image
    denoised_recon = denoise(recon_val)

    # Calculate PSNR on CS reconstructed and denoised image
    bitrate_cs, distortion_cs = estimate_bitrate_and_distortion(denoised_recon, im1, lambda_factor)
    print("CS BPP:", bitrate_cs)
    print("CS PSNR:", distortion_cs)

    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)

    with open(outputfolder + 'quantized_res_feature.pkl', 'wb') as output:
        pickle.dump(Res_q, output)

    with open(outputfolder + 'quantized_res_prior_feature.pkl', 'wb') as output:
        pickle.dump(Res_prior_q, output)

    with open(outputfolder + 'quantized_motion_feature.pkl', 'wb') as output:
        pickle.dump(motion_q, output)

    with open(outputfolder + 'quantized_measurements.pkl', 'wb') as output:
        pickle.dump(quantized_measurements, output)

    with open(outputfolder + 'measurement_matrix.pkl', 'wb') as output:
        pickle.dump(measurement_matrix, output)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--EncoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_E.pb', help="encoder model")
    parser.add_argument('--input_frame', type=str, dest="input_path", default='./frame', help="input frame folder")
    parser.add_argument('--refer_frame', type=str, dest="refer_path", default='./frame', help="refer frame folder")
    parser.add_argument('--outputpath', type=str, dest="outputfolder", default='./testpkl/', help="output pkl folder")
    parser.add_argument('--num_measurements', type=int, dest="num_measurements", default=100, help="number of random measurements")

    args = parser.parse_args()

    input_folder = args.input_path
    refer_folder = args.refer_path

    input_images = sorted(os.listdir(input_folder))
    refer_images = sorted(os.listdir(refer_folder))

    for input_image, refer_image in zip(input_images, refer_images):
        input_image_path = os.path.join(input_folder, input_image)
        refer_image_path = os.path.join(refer_folder, refer_image)

        encoder(args.loadmodel, input_image_path, refer_image_path, args.outputfolder, args.num_measurements)
