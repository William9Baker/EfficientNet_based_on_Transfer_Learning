# --coding:utf-8--
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
# from keras.models import load_model
from PIL import Image

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects



get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})


width = 150
height = 150

# dog_img= dog_images[-1]
img = './12499.jpg'
# Image(filename=dog_img)

def predict_image(img_path):
    # Read the image and resize it
    img = image.load_img(img_path, target_size=(height, width))
    # Convert it to a Numpy array with target shape.
    x = image.img_to_array(img)
    # Reshape
    x = x.reshape((1,) + x.shape)
    x /= 255.
    print(type(x))   # <class 'numpy.ndarray'>
    # result = model.predict([x])[0][0]
    result = model.predict([x])  # 和result = model.predict(x) 输出的结果一样
    if result > 0.5:
        animal = "cat"
    else:
        animal = "dog"
        result = 1 - result
    return animal, result


# 载入模型
# model = load_model('./output_model_file/my_model.h5')
model = load_model('./output_model_file5/my_model_5.h5')

# img = Image.open('12499.jpg')
# img_path = './12499.jpg'

# print(predict_image(cat_img))
print(predict_image(img))
# print(predict_image(img))
