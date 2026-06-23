# Use an official NVIDIA image with Python and PyTorch pre-installed
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /workspace

# Copy the requirements file first to optimize Docker layer caching
COPY requirements.txt .

# Install dependencies (KeyBERT, pandas, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project structure into the container
COPY . .

# Set default environment variable to force PyTorch to use GPU if available
ENV FORCE_CUDA=1

# Command to launch the Streamlit app when the container starts
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]