import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(layout="wide", page_title="Welcome, I am your personal Dog classifier.")

st.write("## Upload photo of your dog and lets us tell you the breed")

st.sidebar.write("## Upload and download :gear:")

learn = load_learner("model.pkl")

categories = ("German-shephard","Labrador")

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.sidebar.write("## Upload and download :gear:", classify_image(my_upload))

# is_dog,_,probs = learn.predict('labrador.jpg')
# print(f"This is a: {is_dog}.")
# print(f"Probability it's a {is_dog}: {probs[0]:.4f}")
# Image.open('labrador.jpg').to_thumb(256,256)
