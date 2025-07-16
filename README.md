# FAIR NeXus GUI Assistant

🔬 A lightweight and user-friendly Streamlit app to assist researchers in generating **FAIR NeXus metadata** files from microscope images — integrated with local **LLMs running via Ollama** on the ORFEO cluster.

---

## ✨ Features

- 📤 Upload microscope images (`.tif`, `.png`, `.jpg`)
- 🧾 Fill in metadata fields interactively (instrument, sample ID, etc.)
- 🧠 Use LLM (e.g., `llama3`, `deepseek-coder`) via Ollama API to:
  - Suggest NeXus application definition
  - Generate NeXus metadata in Python
- 💾 View and download NeXus file
- 🌐 Optional: Upload NeXus file to SFTP/HTTP server

---

## ⚙️ Requirements

- **Python 3.10**
- [`ollama`](https://ollama.com/) installed and running (locally or via SSH tunnel)
- All dependencies listed in [`requirements.txt`](./requirements.txt)

---

### 🌐 Environment Configuration

This app connects to an **Ollama API server** (e.g., for LLMs like `llama3`, `deepseek-coder`).

To configure the connection, create a `.env` file in the project root:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434

## 🚀 Usage

## Quick start (local)

```bash

# Clone
git clone https://github.com/elahehsaadat/fair-nexus-gui
cd fair-nexus-gui

# Create virtual environment (Python 3.10)
py -3.10 -m venv .venv && source .venv/bin/activate    # or Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt (pip install --only-binary=:all: -r requirements.txt)


# Tunnel to ORFEO (if using remote LLM)
access to ollama on orfeo "ssh -L 11434:10.128.2.165:11434 orfeo"
echo "OLLAMA_BASE_URL=http://localhost:11434" > .env # adjust for remote

# Run the app
streamlit run nexus_gui.py
