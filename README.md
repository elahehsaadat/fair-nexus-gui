# FAIR NeXus File Metadata Assistant

A Streamlit app for generating FAIR-compliant NeXus files from microscope image data and metadata. Supports local Ollama models.

## Features
- Upload .tif, .png, .jpg images
- Input experimental metadata
- Query local LLM (Ollama)
- Generate `.nxs` file using `nexusformat`
- Download & preview NeXus tree

## Requirements

```bash
pip install -r requirements.txt
