import gradio as gr
from fastai.vision.all import *

learn = load_learner("model.pkl")

categories = ("German-shephard","Labrador")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['labrador.jpg','gsd.jpg','bully.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)