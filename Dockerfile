# Use an official Python runtime as a parent image. 'slim' is smaller.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the file that lists your Python dependencies
COPY requirements.txt .

# Install the dependencies. --no-cache-dir keeps the image size down.
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (app.py, .h5 model, html, etc.) into the container
COPY . .

# --- Security Best Practice: Run as a non-root user ---
RUN addgroup --system app && adduser --system --group app

# Create and set permissions for the directory where images will be saved
RUN mkdir -p /app/uploads && chown -R app:app /app/uploads
USER app

# Expose port 5000 to allow communication with the app
EXPOSE 5000

# Command to run the application using Gunicorn, a production-ready web server.
# The timeout is increased to 120s to give the model enough time to make a prediction.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "--preload", "app:app"]