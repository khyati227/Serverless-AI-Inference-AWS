# Serverless-AI-Inference-AWS
Performance and cost analysis of serverless AI inference pipelines using AWS Lambda, Docker, and PyTorch

Experimental Setup
The inference pipeline was tested using multiple AWS Lambda memory configurations:
512 MB
1024 MB
2048 MB
3008 MB

Both cold-start and warm-start invocations were analyzed to understand how initialization overhead and allocated resources affect inference performance.
Experimental results are available in analyze-results/results/

Running the Project Locally
1. Install Dependencies
pip install -r requirements.txt
2. Build the Docker Image
docker build -t serverless-ai-inference .
3. Run the Lambda Container Locally
docker run -p 9000:8080 serverless-ai-inference

Results
The collected experimental data demonstrates how serverless AI inference performance changes based on factors including:

Lambda memory allocation
Cold vs. warm execution
Model initialization overhead
Compute resources available to the Lambda function
