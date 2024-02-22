from flask import Flask, render_template, request, url_for
import os
from glob import glob
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.applications import imagenet_utils


app = Flask(__name__, static_url_path='/static')

# Load ResNet50V2 model
resnet50v2 = ResNet50V2(weights='imagenet')

# Manually map ImageNet class names to your dataset clas
class_name_mapping = {
    'n07873807': 'pizza',
    'n07753592': 'banana',
    'n02504458': 'African_elephant',
    'n03291819': 'table',
    'n02979186': 'camera',
    'n03793489':'mouse',
    'n03196217':'digital_clock',
    'n04548280':'wall_clock',
    'n03197337':'digital_watch',
    'n04356056':'sunglasses',
    'n03085013':'computer_keyboard',
    'n03180011':'desktop_computer',
    'n04355933':'sunglass',
    'n02948072':'candle',
    'n03594734':'jean',
    'n02999410':'chain',
    'n04120489':'running_shoe',
    'n04370456':'sweatshirt',
    'n03617480':'kimono',
    'n03916031':'perfume',
    'n04069434':'reflex_camera',
    'n03976467':'Polaroid_camera',
    'n04350905':'suit',
    'n03047690':'clog',
    'n03595614':'jersey',
    'n04264628':'space_bar'

}

def predict_image_class(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    resized_img = np.expand_dims(img, axis=0)
    final_image = preprocess_input(resized_img)

    predictions = resnet50v2.predict(final_image)
    results = imagenet_utils.decode_predictions(predictions)

    predicted_class, _, _ = results[0][0]

    predicted_class_folder = 'dataset/' + class_name_mapping.get(predicted_class, '').lower()
    class_images = glob(predicted_class_folder + '/*.jpg')
    class_images = [os.path.basename(image_path) for image_path in class_images]

    return predicted_class_folder, class_images

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/both.html', methods=['GET', 'POST'])
def both():
    predicted_class_folder = None
    class_images = None

    if request.method == 'POST':

        file = request.files['file']
        if file:
            
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            
            predicted_class_folder, class_images = predict_image_class(file_path)

           
            return render_template('results.html', file_path=file_path, class_images=class_images, predicted_class_folder=predicted_class_folder)

    return render_template('both.html')



if __name__ == '__main__':
    app.run(debug=True)
