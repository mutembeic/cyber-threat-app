# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

# Set an environment variable to tell NLTK where to store/find its data
ENV NLTK_DATA=/code/nltk_data

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- NLTK DATA FIX (BRUTE FORCE METHOD) ---
# 1. Explicitly create the NLTK data directory
RUN mkdir -p $NLTK_DATA
# 2. Use the python3 executable to run the downloader and specify the download directory
RUN python3 -m nltk.downloader -d $NLTK_DATA wordnet stopwords punkt punkt_tab

# Copy the application code and model
COPY backend/ /code/app
COPY model/ /code/model

# Change ownership of the entire /code directory to ensure writability
RUN chown -R 1000:1000 /code

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]