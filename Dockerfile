# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the backend application code
COPY backend/ /code/app

# Copy the model folder
COPY model/ /code/model

# --- FIX FOR PERMISSION ERROR ---
# 1. Create the nltk_data directory as the root user during the build
RUN mkdir -p /code/app/nltk_data
# 2. Change the ownership of the entire /code directory to the default non-root user
RUN chown -R 1000:1000 /code

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]