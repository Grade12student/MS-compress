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


    # Preprocess the input_image
    # For example, you might need to normalize the values or expand the dimensions
    input_image_processed = preprocess_input_image(input_image)

    # Fit the model to your data
    model.fit(x=input_image_processed, y='/frame/', epochs=10, batch_size=32)  # Adjust your_target_data as needed

    # Use the model to predict the high-resolution output
    predicted_high_res_output = model.predict(input_image_processed)

    return predicted_high_res_output  # Return the predicted high-resolution output


def preprocess_input_image(input_image):
    # Normalize the values to a range of 0 to 1
    input_image_processed = input_image.astype('float32') / 255.0

    # Expand the dimensions to match the expected shape of the CNN model
    input_image_processed = np.expand_dims(input_image_processed, axis=0)

    # Return the preprocessed input_image
    return input_image_processed

def preprocess_target_image(target_image):
    # Preprocess the target image before feeding it to the model
    target_image_resized = cv2.resize(target_image, (416, 224), interpolation=cv2.INTER_LINEAR)
    target_image_processed = target_image_resized / 255.0  # Normalize pixel values to [0, 1]
    target_image_processed = np.expand_dims(target_image_processed, axis=0)  # Add an extra dimension to match model input
    return target_image_processed

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)  # Assuming 3 output channels (RGB)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def decoder(model, loadmodel, refer_frame, outputfolder, output_frame):
    graph = load_graph(loadmodel)
    sess = tf.compat.v1.Session(graph=graph)  # Initialize the TensorFlow session

    reconframe = graph.get_tensor_by_name('import/build_towers/tower_0/train_net_inference_one_pass/train_net/ReconFrame:0')
    res_input = graph.get_tensor_by_name('import/quant_feature:0')
    res_prior_input = graph.get_tensor_by_name('import/quant_z:0')
    motion_input = graph.get_tensor_by_name('import/quant_mv:0')
    previousImage = graph.get_tensor_by_name('import/input_image_ref:0')

    residual_feature = []  # Initialize the variable to avoid any potential issues

    if os.path.isfile(refer_frame):
        # If the refer_frame argument is a file, process the single frame
        refer_image = cv2.imread(refer_frame)
        if refer_image is not None:
            # Resize refer_image
            low_res = cv2.resize(refer_image, (416, 224), interpolation=cv2.INTER_LINEAR)
        else:
            print("Failed to load the reference image.")
            return

        refer_image = refer_image / 255.0
        refer_image = np.expand_dims(refer_image, axis=0)
        output_folder = output_frame

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # ... (rest of the code remains the same)

    elif os.path.isdir(refer_frame):
        # If the refer_frame argument is a directory, process all frames in the directory
        i = 0  # Initialize a counter for incrementing filenames
        for filename in os.listdir(refer_frame)[:5]:  # Testing on a small number of frames
            print("Processing", filename)
            refer_image = cv2.imread(os.path.join(refer_frame, filename))
            if refer_image is None:
                print(f"Failed to load the reference image from {os.path.join(refer_frame, filename)}")
                continue
            
            if len(residual_feature) > i:  # Check if the index is within bounds of residual_feature
                if residual_feature[i].shape[:2] != (224, 416):
                    resized_residual_feature = cv2.resize(residual_feature[i], (224, 416), interpolation=cv2.INTER_LINEAR)
                else:
                    resized_residual_feature = residual_feature[i]
                
            # Process refer_image before resizing
            # Resize refer_image
            low_res = cv2.resize(refer_image, (416, 224), interpolation=cv2.INTER_LINEAR)

            refer_image = refer_image / 255.0
            refer_image = np.expand_dims(refer_image, axis=0)
            output_folder = output_frame

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Preprocess target image for training
            target_image_path = os.path.join(refer_frame, filename)
            target_image = cv2.imread(target_image_path)
            target_image_processed = preprocess_target_image(target_image)  # Preprocess the target image

            input_image_processed = preprocess_input_image(low_res)  # Preprocess the input image

            # Fit the model with the preprocessed data
            model.fit(x=input_image_processed, y=target_image_processed, epochs=10, batch_size=32)

            i += 1  # Increment the counter for the next filename

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

            if residual_feature is not None and len(residual_feature) > i:   # Check if residual_feature is not None and the index is within bounds
                if residual_feature[i].shape[:2] != (224, 416):
                    resized_residual_feature = cv2.resize(residual_feature[i], (224, 416), interpolation=cv2.INTER_LINEAR)
                else:
                    resized_residual_feature = residual_feature[i]

                print("Shapes of low_res and resized_residual_feature:", low_res.shape, resized_residual_feature.shape)  # Add this line to print the shapes of the arrays

                # Ensure the ordering of dimensions is appropriate
                if len(resized_residual_feature.shape) == 3:
                    resized_residual_feature = np.transpose(resized_residual_feature, (1, 0, 2))  # Adjust dimension ordering if necessary

                # Ensure the shapes of low_res and resized_residual_feature are compatible
                if low_res.shape != resized_residual_feature.shape:
                    resized_residual_feature = cv2.resize(resized_residual_feature, (low_res.shape[1], low_res.shape[0]))

                resized_residual_feature = resized_residual_feature.sum(axis=-1, keepdims=True)

                # Convert low_res to float32 to match the data type of resized_residual_feature
                low_res = low_res.astype(np.float32)

                # Perform addition after ensuring the data types are compatible
                low_res = np.add(low_res, resized_residual_feature, out=low_res)

                # Map to high-resolution using a CNN
                high_res = custom_cnn_function(low_res)  # Replace custom_cnn_function with your own CNN implementation

                # Convert the high-resolution output to the appropriate data type for saving the image
                high_res = (high_res * 255).astype(np.uint8)

                # Save the output with an incremented filename
                output_path = os.path.join(output_folder, f"decoded_{i}.png")
                
                cv2.imwrite(output_path, high_res)
                print(f"Saved decoded image at: {output_path}")

                i += 1  # Increment the counter for the next filename

    else:
            raise ValueError(f"Invalid input provided for refer_frame: {refer_frame}")

    sess.close()  # Close the TensorFlow session

    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--DecoderModel', type=str, dest="loadmodel", default='./model/L2048/frozen_model_D.pb', help="decoder model")
    parser.add_argument('--refer_frame', type=str, dest="refer_frame", default='./E/', help="refer frame path")
    parser.add_argument('--loadpath', type=str, dest="outputfolder", default='./E/', help="saved pkl file")
    parser.add_argument('--output_frame', type=str, dest="output_frame", default='./D/', help="output frame directory")

    args = parser.parse_args()

    input_shape = (224, 416, 3)  # Define the input shape here

    model = create_model(input_shape)  # Create the model here

    decoder(model, **vars(args))  # Pass the 'model' as an argument to the decoder function



