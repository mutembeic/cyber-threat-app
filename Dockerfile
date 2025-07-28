# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /code

# Copy the backend requirements file and install dependencies
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your application code into the container
# This includes the backend code and the model files
COPY ./backend /code/app
COPY ./model /code/app/model

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
# The --host 0.0.0.0 makes the server accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]