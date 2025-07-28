# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- NLTK DATA FIX (DEFINITIVE) ---
# 1. Create the local data directory
RUN mkdir -p /code/app/nltk_data
# 2. Set an environment variable to tell NLTK to ALWAYS use this path
ENV NLTK_DATA /code/app/nltk_data
# 3. Download ALL required packages directly into that path
RUN python -m nltk.downloader -d /code/app/nltk_data wordnet stopwords punkt punkt_tab

# Copy the backend application code
COPY backend/ /code/app

# Copy the model folder
COPY model/ /code/model

# Change ownership of the entire app directory to ensure writability if needed later
# Although not strictly necessary with the ENV var, it's good practice.
RUN chown -R 1000:1000 /code/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]