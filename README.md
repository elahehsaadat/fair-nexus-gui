# FAIR NeXus GUI Assistant

A Streamlit application for extracting microscopy-image metadata, mapping it to
NeXus definitions, and generating FAIR NeXus files. Model-assisted features use
the ORFEO LLM gateway.

## Features

- Upload TIFF, PNG, or JPEG microscopy images.
- Extract embedded TIFF metadata or load a separate JSON header.
- Select a NeXus application definition and map metadata fields.
- Generate, validate, and download a NeXus file.
- Use an ORFEO-hosted language model to assist with metadata interpretation.

## Requirements

- Python 3.10–3.12
- Git, including Git submodule support
- ORFEO credentials for model-assisted features

## Setup

Clone the repository and its NeXus definitions submodule:

```bash
git clone --recurse-submodules https://github.com/elahehsaadat/fair-nexus-gui.git
cd fair-nexus-gui
```

For an existing clone, initialize the submodule with:

```bash
git submodule update --init --recursive
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy the example environment file and replace the placeholder credentials:

```bash
cp .env.example .env
```

The application reads these variables:

```dotenv
ORFEO_AUTH_URL=https://orfeo-auth.areasciencepark.it/application/o/token/
ORFEO_LLM_BASE_URL=https://orfeo-llm.areasciencepark.it/vllm
ORFEO_CLIENT_ID=your-client-id
ORFEO_USERNAME=your-username
ORFEO_PASSWORD=your-password
```

The local `.env` file is ignored by Git. Never commit real credentials.

## Run

```bash
python -m streamlit run nexus_gui.py
```

Open <http://localhost:8501> if Streamlit does not open a browser automatically.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

## Docker

```bash
docker build -t fair-nexus-gui .
docker run --rm -p 8501:8501 --env-file .env fair-nexus-gui
```

## Project layout

- `nexus_gui.py`: Streamlit interface and workflow
- `nexus_generator.py`: NeXus file generation and validation
- `upload_utils.py`: image, metadata, ORFEO, and upload helpers
- `mapping_store.py`: saved mapping persistence
- `nexus_definitions/`: pinned NeXus definitions Git submodule
- `tests/`: regression tests

## License

This project is distributed under the [MIT License](LICENSE).
