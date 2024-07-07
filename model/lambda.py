import boto3
import json
import io
import urllib.parse
import numpy as np
from PIL import Image

s3_client = boto3.client("s3")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# SageMaker endpoint name
ENDPOINT_NAME = "tensorflow-inference-2024-07-07-20-41-57-262"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def preprocess_image(image_content):
    img = Image.open(image_content)
    img = img.resize((150, 150))
    img = np.array(img)

    # Ensure that the image has three channels (for RGB images)
    if img.shape[-1] != 3:
        img = img[:, :, :3]  # Keep only the first three channels

    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def lambda_handler(event, context):
    # Get the bucket name and key for the uploaded S3 object
    bucket = event["Records"][0]["s3"]["bucket"]["name"]

    key = urllib.parse.unquote_plus(
        event["Records"][0]["s3"]["object"]["key"], encoding="utf-8"
    )

    extension = key.rsplit(".", 1)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return {"statusCode": 400, "body": json.dumps("Invalid file type")}

    # Retrieve the uploaded file content
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_data = response["Body"].read()
    image_content = io.BytesIO(file_data)
    input_data = preprocess_image(image_content)

    # Send the file content to the SageMaker endpoint for prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="image/" + extension,
        Body=input_data.tobytes(),
    )

    # Process the prediction result
    result = json.loads(response["Body"].read().decode())
    label = "GOOD WELD" if result > 0.8 else "BAD WELD"
    print("Prediction result:", label)

    return {"statusCode": 200, "body": json.dumps("Prediction completed successfully!")}
