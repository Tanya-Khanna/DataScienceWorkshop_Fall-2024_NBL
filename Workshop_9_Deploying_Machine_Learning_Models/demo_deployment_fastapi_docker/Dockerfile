# Dockerfile

# Use an official Python image as the base for the container.
# The "python:3.11-slim" image provides a lightweight version of Python 3.11,
# reducing the final image size by removing unnecessary tools and libraries.
# This makes it efficient for deployment, especially in production environments.
FROM python:3.11-slim 

# Set the working directory inside the container to "/app".
# All subsequent commands (such as copying files or installing packages)
# will be executed within this directory, keeping the project structure organized.
# This directory is also where our application code will reside in the container.
WORKDIR /app

# Update package lists to ensure we have access to the latest versions from repositories
RUN apt-get update && \
    # Install necessary build tools, excluding recommended extras to keep the image lightweight
    apt-get install -y --no-install-recommends \
    build-essential \
    # Remove cached package lists to reduce the image size
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
# This step copies the requirements.txt file from the host machine to the current directory in the container.
# requirements.txt lists all the dependencies our FastAPI application needs to run.
COPY requirements.txt .

# Install dependencies listed in requirements.txt
# The "RUN" command executes commands within the container. Here, we use pip to install dependencies.
# The "--no-cache-dir" flag prevents pip from caching the downloaded packages, reducing image size.
# This step ensures that all necessary libraries are installed and available for the application.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code and model file into the container
# The following COPY commands transfer the main application file (app.py) and the pre-trained model (iris_classifier.pkl)
# from the host system to the container. These files are essential for running the FastAPI application and making predictions.
# Copying these files into the container allows the FastAPI app to access the model and serve requests.
COPY iris_classifier.pkl .
COPY app.py .

# Expose port 8000 for the FastAPI app
# The EXPOSE command tells Docker that the container will listen on port 8000.
# This port is typically used for FastAPI applications and allows external requests to reach the app.
# When the container runs, users can access the API by connecting to this port.
EXPOSE 8000

# Run the FastAPI app with uvicorn (a high-performance server for FastAPI)
# The CMD instruction specifies the command to execute when the container starts.
# Here, we use "uvicorn" to run the FastAPI application defined in "app.py".
# 
# - "app:app" refers to the FastAPI instance named "app" in the "app.py" file.
# - "--host", "0.0.0.0" configures the server to listen on all network interfaces,
#   making it accessible both inside and outside the container.
# - "--port", "8000" specifies that the application will run on port 8000.
#
# This command launches the FastAPI application with uvicorn when the container starts, 
# making it ready to handle requests at the specified host and port.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
