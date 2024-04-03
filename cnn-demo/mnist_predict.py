from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# load model
model = load_model('mnist.h5')


# test with new images


for i in range(0,10):
    file = f'{i}.jpg'
    image = load_img(file, color_mode='grayscale', target_size=(28, 28))
    image = img_to_array(image)
    image = 255 - image # invert image
    # treshold = 128
    # image[image > treshold] = 255
    # image[image <= treshold] = 0
    plt.imshow(image)
    plt.show()
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image /= 255.0    
    pred_new_proba = model.predict(image)
    pred_new = pred_new_proba.argmax(axis=1)
    print(f'pred: {pred_new[0]}, real: {file}')