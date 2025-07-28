# Text-Based Cyber Threat Detector

This project is a full-stack machine learning application designed to classify cybersecurity-related text as either **Malicious** or **Benign**. It leverages Natural Language Processing (NLP) and deep learning to analyze unstructured text from sources like threat intelligence reports, emails, or system logs.

The project demonstrates an end-to-end workflow, from data exploration and model training in a Jupyter Notebook to a deployed, interactive web application.

---

### Live Application Links

*   **. Live Web App (Streamlit):** [`<your-streamlit-app-url>`](https://<your-streamlit-app-url>)
*   **. Live API (Hugging Face):** [`<your-hugging-face-space-url>`](https://<your-hugging-face-space-url>)
*   **. API Documentation (Swagger):** [`<your-hugging-face-space-url>/docs`](https://<your-hugging-face-space-url>/docs)

---

## Features

-   **Deep Learning Model:** Utilizes a Keras/TensorFlow model to perform binary text classification.
-   **RESTful API:** A robust backend built with FastAPI to serve the model and handle prediction requests.
-   **Interactive Frontend:** A user-friendly web interface created with Streamlit that allows users to input text and receive instant analysis.
-   **Containerized Deployment:** The backend is containerized with Docker for consistent and scalable deployment.
-   **CI/CD Ready:** Hosted on cloud platforms (Hugging Face and Streamlit Cloud) that integrate directly with GitHub.

## Technology Stack

-   **Model Development:** Python, Pandas, NLTK, Scikit-learn, TensorFlow/Keras
-   **Backend:** FastAPI, Uvicorn
-   **Frontend:** Streamlit, Requests
-   **Deployment:** Docker, Hugging Face Spaces (Backend), Streamlit Community Cloud (Frontend)
-   **Code Hosting:** Git & GitHub

## Project Structure

The repository is organized to separate the application code from the analysis and research materials.

```
.
├── .gitignore
├── README.md
│
├── analysis/               # Research, data, and presentation files
│   ├── data/
│   │   └── cyber-threat-intelligence_all.csv
│   └── notebooks/
│       └── final-notebook.ipynb
│
├── backend/                # FastAPI application code
│   ├── main.py
│   └── requirements.txt
│
├── frontend/               # Streamlit application code
│   ├── frontend.py
│   └── requirements.txt
│
├── model/                  # The final, trained model file
│   └── improved_base_model.h5
│
└── Dockerfile              # Configuration for the backend container
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- `pip` package manager
- Git and Git LFS (`sudo apt-get install git-lfs`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  **Install Git LFS and pull the model file:**
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies for both frontend and backend:**
    ```bash
    pip install -r backend/requirements.txt
    pip install -r frontend/requirements.txt
    ```

## How to Run Locally

You need to run the backend and frontend in two separate terminal windows (with the virtual environment activated in both).

### 1. Start the FastAPI Backend

In your first terminal, from the project's root directory:
```bash
uvicorn backend.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 2. Start the Streamlit Frontend

In your second terminal, from the project's root directory:
```bash
streamlit run frontend/frontend.py
```
The web application will open in your browser at `http://localhost:8501`.

## Authors

This project was a collaborative effort by:
- Kelvin Kipkorir
- Lucy Mutua
- Charles Mutembei
- Sharon Aoko
- Victor Musyoki