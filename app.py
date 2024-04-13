import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

from flask import Flask, jsonify, request
import numpy as np
import array

from keras.models import load_model
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import json
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from tensorflow import keras
from tensorflow.keras import layers

from keras.initializers import Initializer
import random

num_channels = 1  # Assuming RGB images
input_shape = (16, 16, 1)

latent_dim = 10
# Encoder
input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
conv_shape = tf.keras.backend.int_shape(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
z_mu = Dense(latent_dim, name='latent_mu')(x)
z_sigma = Dense(latent_dim, name='latent_sigma')(x)

def sample_z(args):
    z_mu, z_sigma = args
    eps = tf.random.normal(shape=(tf.shape(z_mu)[0], latent_dim), mean=0., stddev=1.)
    return z_mu + tf.exp(z_sigma / 2) * eps

z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mu, z_sigma])
encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')

# Decoder
decoder_input = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(decoder_input)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)
decoder = Model(decoder_input, x, name='decoder')

# Apply the decoder to the latent sample
z_decoded = decoder(z)


# Define the PixelConvLayer class
class PixelConvLayer(layers.Layer):
    def __init__(self, filters, kernel_size, mask_type='B', **kwargs):
        super().__init__(**kwargs)
        self.mask_type = mask_type
        self.conv = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = tf.shape(self.conv.kernel)
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 1, activation="relu")
        self.pixel_conv = PixelConvLayer(filters, kernel_size)
        # self.conv2 = layers.Conv2D(filters, 1, activation="relu")

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.pixel_conv.build(input_shape)
        # self.conv2.build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        # x = self.conv2(x)
        return layers.add([inputs, x])


# Define the custom layer class
class CustomLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss (as we used sigmoid activation we can use binary crossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)

        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)

# Create an instance of the CustomLayer class
custom_layer = CustomLayer()

# Apply the custom layer to the input images and the decoded latent distribution sample
vae_loss = custom_layer([input_img, z_decoded])

# Define and summarize the VAE model
vae = Model(input_img, input_img, name='vae')  # Output should be the same as input for VAE
vae.compile(optimizer='adam')  # No need to specify loss=None since we added the loss in the custom layer


# Define the input size
input_size = 100

# Define the threshold for binarization
threshold = 0.5

app = Flask(__name__)

# Load the VAE room model
loaded_room_vae_model = tf.keras.models.load_model("room_vae_model.h5", custom_objects={'CustomLayer': CustomLayer})

# Load the GAN room model
loaded_room_gan_model = tf.keras.models.load_model("room_gan_model.h5")

# Get the roomdecoder model from the loaded VAE model
room_decoder_model = loaded_room_vae_model.get_layer('decoder')

print("Loaded room model!")

# Define a route to generate dungeon data
@app.route('/dungeon_gan', methods=['GET'])
def generate_dungeon_gan():

    # Load the Keras model
    loaded_gan_model = tf.keras.models.load_model("dungeon_gan_model.h5")
    
    # Generate random noise for input to the generator part of the GAN
    noise = tf.random.normal((1, input_size))

    # Generate a single dungeon sample using the generator part of the GAN model
    generated_dungeon = loaded_gan_model.layers[1](noise)

    # Apply threshold to binarize the sample
    binarized_dungeon = (generated_dungeon > threshold).numpy().astype(int)

    # Reshape binarized dungeon to match desired format (if needed)
    reshaped_dungeon = binarized_dungeon.reshape(8, 8)

    # Convert the generated dungeon to a JSON-like array in the desired format
    generated_dungeon_json = []

    for row in reshaped_dungeon:
        generated_dungeon_json.append(row.tolist())

    # Return the JSON-like representation of the generated dungeon
    return jsonify(generated_dungeon_json)

@app.route('/dungeon_pixelcnn', methods=['GET'])
def generate_dungeon_pixelcnn():
    # Load the pixelcnn_model model with custom objects
    loaded_model = tf.keras.models.load_model('dungeon_pixelcnn_model.h5', custom_objects={'PixelConvLayer': PixelConvLayer, 'ResidualBlock': ResidualBlock})

    # Load dungeon data from numpy file
    data = np.load('dungeons_dataset.npy')

    # Reshape the data to match the desired shape (8, 8, 1)
    data = data.reshape(-1, 8, 8, 1)

    # Scale the pixel values to [0, 1] range
    data = data.astype("float32") / 1.0  # Assuming values are already either 0 or 1

    # Randomly select a dungeon from the dataset
    random_index = random.randint(0, len(data) - 1)
    random_dungeon = data[random_index:random_index + 1]

    # Generate a single dungeon using the loaded model
    generated_dungeon = loaded_model.predict(random_dungeon)

    # Apply threshold to binarize the sample
    threshold = 0.5
    binarized_dungeon = (generated_dungeon > threshold).astype(int)

    # Reshape binarized dungeon to match desired format (if needed)
    reshaped_dungeon = binarized_dungeon.reshape(8, 8)

    # Convert the generated dungeon to a JSON-like array
    generated_dungeon_json = reshaped_dungeon.tolist()

    # Return the JSON-like representation of the generated dungeon
    return jsonify(generated_dungeon_json)

@app.route('/get_dungeon_data', methods=['GET'])
def get_dungeon_data():

    # Load dungeon data (adjust the file path if needed)
    data = np.load('dungeons_dataset.npy')

    # Select the first dungeon
    selected_dungeon = data[0]

    # Convert the selected dungeon from NumPy array to list
    dungeon_list = selected_dungeon.tolist()

    return jsonify(dungeon_list)

@app.route('/get_room_data', methods=['GET'])
def get_room_data(): 
    # Load room data (adjust the file path if needed)
    room_data = np.load('rooms_dataset.npy')

    # Select the first room
    selected_room = room_data[0]

    # Convert the selected room from NumPy array to list
    room_list = selected_room.tolist()

    return jsonify(room_list)

# Define a route to generate room data using VAE
@app.route('/room_vae', methods=['GET'])
def generate_room_vae():
    # Generate a random latent vector for input to the decoder part of the VAE
    latent_dim = 10
    random_latent_vector = np.random.normal(size=(1, latent_dim)).astype(np.float32)  # Convert to float32

    # Generate a single room sample using the decoder part of the VAE model
    generated_room = room_decoder_model.predict(random_latent_vector)

    # Multiply the values by 1000 to move the decimal point
    generated_room_scaled = generated_room * 1000

    # Apply the custom rounding function to the generated room
    rounded_generated_room_custom = np.vectorize(custom_round)(generated_room_scaled)

    # Reshape the generated room to 16x16
    reshaped_generated_room = rounded_generated_room_custom.reshape((16, 16))

    # Convert the rounded room to a JSON-like array
    rounded_generated_room_json = []

    for row in reshaped_generated_room:
        rounded_generated_room_json.append(row.tolist())

    # Return the JSON-like representation of the generated room
    return jsonify(rounded_generated_room_json)

# Define a route to generate room data using GAN
@app.route('/room_gan', methods=['GET'])
def generate_room_gan():
    # Generate random noise for input to the generator part of the GAN
    noise = tf.random.normal((1, input_size))

    # Generate a single room sample using the generator part of the GAN model
    generated_room = loaded_room_gan_model.layers[1](noise)

    # Convert the generated room to a NumPy array to access its shape
    generated_room_numpy = generated_room.numpy()

    # Get the grid size
    grid_size = int(np.sqrt(generated_room_numpy.shape[1]))

    # Reshape the generated room to match a square grid (assuming it represents an image)
    generated_room_reshaped = tf.reshape(generated_room, (-1, grid_size, grid_size))

    # Rounding
    # Multiply the values by 1000 to move the decimal point
    generated_room_scaled = generated_room_reshaped * 1000

    # Apply the custom rounding function to the generated room
    rounded_generated_room_custom = np.vectorize(custom_round)(generated_room_scaled)

    # Convert the rounded room to a JSON-like array
    rounded_generated_room_json = []

    for row in rounded_generated_room_custom[0]:
        rounded_generated_room_json.append(row.tolist())

    # Return the JSON-like representation of the generated room
    return jsonify(rounded_generated_room_json)




# Define your custom rounding function
def custom_round(value):
    if value <= 0.5:
        return 0
    elif value > 0.5 and value <= 1.5:
        return 1        
    elif value > 1.5 and value <= 2.5:
        return 2
    elif value > 2.5 and value <= 3.5:
        return 3        
    elif value > 3.5 and value <= 4.5:
        return 4
    elif value > 4.5 and value <= 5.5:
        return 5        
    elif value > 5.5 and value <= 6.5:
        return 6
    elif value > 6.5 and value <= 7.5:
        return 7  
    elif value > 7.5 and value <= 8.5:
        return 8              
    else:
        return 0    


if __name__ == '__main__':
    # Run the app on the server's publicly available IP address, on port 8080
    app.run(host='0.0.0.0', port=8080)
