#!/usr/bin/env python
# coding: utf-8

import os
import io
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import pipeline

####################################
########## YOUR CODE HERE ##########
####################################
# You will need to download a model
# to implement summarization from 
# HugginFace Hub.
#
# You may want to use following models:
# https://huggingface.co/Falconsai/text_summarization
# https://huggingface.co/knkarthick/MEETING_SUMMARY
# ...or any other you like, but think of 
# the size of the model (<1GB recommended)
#
# Your code may look like this:
#from transformers import pipeline
#with st.spinner('Please wait, application is initializing...'):
#    MODEL_SUM_NAME = '<YOUR_MODEL>'
#    SUMMARIZATOR = pipeline("summarization", model=MODEL_SUM_NAME)
####################################

with st.spinner('Please wait, application is initializing...'):
    MODEL_SUM_NAME = 'Falconsai/text_summarization'  # Summarization model
    SUMMARIZER = pipeline("summarization", model=MODEL_SUM_NAME)

# page headers and info text

def pdf2img(pdf_bytes):
    """
    Turns pdf file to set of jpeg images.

    """
    images = convert_from_bytes(pdf_bytes.read())
    return images


def ocr_text(img, lang='eng'):
    """
    Takes the text from image.
    
    :lang: language is `eng` by default,
           use `eng+rus` for two languages in document

    """
    text = str(pytesseract.image_to_string(
        img,
        lang=lang
    ))
    return text


def ocr_text_dir(img_dir, lang='eng'):
    """
    Takes the text from images in a folder.

    """
    text = ''
    for img_name in tqdm(sorted(os.listdir(img_dir))):
        if '.jpg' in img_name:
            img = Image.open(f'{IMG_PATH}/{img_name}')
            text_tmp = ocr_text(img, lang=lang)
            text = ' '.join([text, text_tmp])
    return text


st.header('Text OCR from PDF or JPEG', divider='rainbow')

st.write('#### Upload you file or image')
uploaded_file = st.file_uploader('Select a file (JPEG or PDF)')
if uploaded_file is not None:
    file_name = uploaded_file.name
    lang = st.selectbox(
            'Select language to extract ',
            ('eng', 'rus', 'eng+rus')
        )
    if '.jpg' in file_name:
        with st.spinner('Please wait...'):
            bytes_data = uploaded_file.read()
            img = Image.open(io.BytesIO(bytes_data))
            
            # image caption model for uploaded image
            text = ocr_text(img, lang=lang)
            st.divider()
            st.write('#### Text extracted')
            st.write(text)
            with st.spinner('Generating summary...'):
                summary = SUMMARIZER(text, max_length=150, min_length=30, do_sample=False)
                st.divider()
                st.write('#### Summary')
                st.write(summary[0]['summary_text'])            
    elif '.pdf' in file_name:
        with st.spinner('Please wait...'):
            imgs = pdf2img(uploaded_file)
            text = ''
            for img in imgs:
                text_tmp = ocr_text(img, lang=lang)
                text = ' '.join([text, text_tmp])
            st.divider()
            st.write('#### Text extracted')
            st.write(text)
            with st.spinner('Generating summary...'):
                summary = SUMMARIZER(text, max_length=150, min_length=30, do_sample=False)
                st.divider()
                st.write('#### Summary')
                st.write(summary[0]['summary_text'])            
    else:
            st.error('File read error', icon='⚠️')
####################################