import streamlit as st 
from fastai.vision.all import *
from PIL import Image
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Rasmni Klassifikatsiya qilivuchi model")

# rasmni joylash

file = st.file_uploader('Rasmni yuklash', type=['png','jpg','gif','svg','jpeg'])
if file:
    st.image(file)
    # PIL image
    img = PILImage.create(file)
    #model
    model = load_learner('transport_model.pkl')

    #prediction
    prediction, pred_idx, probs = model.predict(img)
    st.success(f"Bashorat: {prediction}")
    st.info(f'Ehtimollik: {probs[pred_idx]*100:.1f}%')
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)