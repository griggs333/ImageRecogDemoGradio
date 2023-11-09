import tensorflow.keras.applications.resnet50
import tensorflow.keras.applications.inception_v3
import tensorflow.keras.applications.xception
import tensorflow.keras.applications.mobilenet_v3
import tensorflow.keras.utils as utils
import numpy as np
import gradio as gr



model_param_dict = {'ResNet50': ['ResNet50', 'resnet50', 224, 224],
                    'InceptionV3': ['InceptionV3', 'inception_v3', 299, 299],
                    'Xception': ['Xception', 'xception', 299, 299],
                    'MobileNetV2': ['MobileNetV2', 'mobilenet_v2', 224, 224]}



def classify(imgfile, model):
    m_img_size_x = model_param_dict[model][2]
    m_img_size_y = model_param_dict[model][3]
    module = getattr(tensorflow.keras.applications, model_param_dict[model][1])
    preprocess = getattr(module, "preprocess_input")
    decode = getattr(module, "decode_predictions")
    model_name = getattr(module, model_param_dict[model][0])
    active_model = model_name(weights='imagenet')


    img = utils.load_img(imgfile, target_size=(m_img_size_x, m_img_size_y))

    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess(img)

    class_predictions = active_model.predict(img)
    decoded_predictions = decode(class_predictions, top=5)[0]
    confidences = {decoded_predictions[i][1]: float(decoded_predictions[i][2]) for i in range(len(decoded_predictions))}

    return confidences

with (gr.Blocks() as demo):

    gr.Markdown("# <center>Image Classifer Demo </center>")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload image", type='filepath')
            model_radio = gr.Radio(label="Model Selection", choices=['ResNet50', 'InceptionV3', 'Xception', 'MobileNetV2'], value='ResNet50')
        with gr.Column():
            prediction_outputs = gr.Label(num_top_classes=5, label="Top Class Predictions")
    with gr.Row():
        # with gr.Column():
            btn = gr.Button(value="Submit", scale=0)
        # with gr.Column():
            clr_bt = gr.ClearButton(scale=0, components=[img_input, prediction_outputs])


    btn.click(fn=classify, inputs=[img_input, model_radio], outputs=prediction_outputs)
# img_input = gr.Image(label="Upload image", type='filepath')
# model_radio = gr.Radio(label="Model Selection", choices=['ResNet50', 'InceptionV3', 'Xception', 'MobileNetV2'], value='ResNet50')
# prediction_outputs = gr.Label(num_top_classes=5)

# gr.Interface(fn=classify, inputs=[img_input, model_radio], outputs=prediction_outputs).launch()
demo.launch()
