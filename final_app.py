from fastai.vision.widgets import *
from fastai.vision.all import *

from pathlib import Path

import streamlit as st

import pathlib

plt = 'Windows'
if plt == 'Windows': 
    pathlib.PosixPath = pathlib.WindowsPath

path = "/app/cv_poc/"

st.set_page_config(layout="wide", page_title="Your Dog Classifier")

st.write("## Check your dog breed with one click")
st.write(
    ":dog: Try uploading an image to check dog breed. Full quality images can be downloaded from the sidebar. :grin:"
)

class Predict:
    def __init__(self):
        self.learn_inference = load_learner(path + "model.pkl")
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        st.sidebar.write("## Upload and download :gear:")
        uploaded_file = st.sidebar.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(400,400), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            try:
                st.success(f"{pred} I am {probs[pred_idx]*100:.0f}% confident.")
                st.caption(f"Caution: I have only been trained on a small set of images. I may also be wrong.")
            except:
                st.write("Sorry, I don't know that Dog")  
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    # file_name='model.pkl'

    predictor = Predict()

