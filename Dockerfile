# Use an official PyTorch runtime as a parent image
FROM python:3.10.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt  # If you have any additional requirements

COPY . .

# Run receipt.py when the container launches
CMD ["python", "receipt.py"]

