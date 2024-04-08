import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

from flask import Flask, jsonify, request
import numpy as np

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

from keras.initializers import Initializer

# # Load the Keras model
# loaded_gan_model = load_model("gan_model.h5")    


# Define the input size
input_size = 100

# Define the threshold for binarization
threshold = 0.5

app = Flask(__name__)


# Define a route to generate dungeon data
@app.route('/output', methods=['GET'])
def generate_dungeon_data():

    # Load the Keras model
    loaded_gan_model = load_model("gan_model.h5")        
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

    # # Convert the generated dungeon to a JSON-like array
    # generated_dungeon_json = json.dumps(reshaped_dungeon.tolist())

    # # Return the JSON-like representation of the generated dungeon
    # return jsonify(generated_dungeon_json)

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


if __name__ == '__main__':
    # Run the app on the server's publicly available IP address, on port 8080
    app.run(host='0.0.0.0', port=8080)
