from flask import Flask, request
import numpy as np
import tensorflow as tf
from PIL import Image

# here create an app object using the constructor for flask
app = Flask(__name__)

model = tf.keras.models.load_model('models/model_3.keras')

# here we impliment the get method
@app.route('/summary', methods=['GET'])
def model_info():
   return {
      "version": "v1",
      "name": "model_3",
      "description": "Classify satellite images of homes that have been affected by a hurricane as damaged or no damage",
      "number_of_parameters": 2601153
   }

# here we define preprocessing methods
def prepareImage(image):
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128,128])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

# here we impliment the post method
@app.route('/inference', methods=['POST'])
def upload_file():
    # check if the post request has the file part
   if 'image' not in request.files:
      # if the user did not pass the image under `image`, we don't know what they are
      # don't, so return an error.
      return '{"error": "Invalid request; pass a binary image file as a multi-part form under the image key."}'
   # get the data
   data = request.files['image']
   binary_file = data.read()
   data = prepareImage(binary_file)
   prob = float(model.predict(data, verbose=0)[0][0])
   if (prob > 0.55):
       return { "prediction": "damage"}
   else:
       return { "prediction": "no_damage"}

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')