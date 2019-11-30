from keras.models import load_model
import keras
from .models import model_from_name


def model_from_checkpoint_path(model_config, latest_weights):
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'], input_width=model_config['input_width'])
    model.load_weights(latest_weights)
    return model


def PSPNet():

    model_config = {
        "input_height": 473,
        "input_width": 473,
        "n_classes": 21,
        "model_class": "pspnet_101",
    }

    model_url = "pspnet101_voc2012.h5"
    latest_weights = keras.utils.get_file("pspnet101_voc2012.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)