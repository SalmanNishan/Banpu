import io
import json
import boto3
import numpy as np
import urllib.parse
from PIL import Image

s3_client = boto3.client("s3")
sagemaker_runtime = boto3.client("sagemaker-runtime")

# SageMaker endpoint name
ENDPOINT_NAME = "tensorflow-inference-2024-07-11-09-00-17-984"
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


def make_prediction(bucket_name, image_key):
    # Check if the file extension is allowed
    extension = image_key.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return {
            "bucket": bucket_name,
            "file": image_key,
            "label": "INVALID FILE TYPE",
            "score": 0,
        }

    # Retrieve the uploaded file content
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    file_data = response["Body"].read()
    image_content = io.BytesIO(file_data)
    input_data = preprocess_image(image_content)

    # Send the file content to the SageMaker endpoint for prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(input_data.tolist()).encode(),
    )

    # Process the prediction result
    response_body = json.loads(response["Body"].read().decode())
    result = response_body["predictions"][0][0]
    label = "GOOD WELD" if result > 0.8 else "BAD WELD"

    return {"bucket": bucket_name, "file": image_key, "label": label, "score": result}


def lambda_handler(event, context):
    results = []

    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"], encoding="utf-8")
        result = make_prediction(bucket, key)
        results.append(result)

    print(results)
    return {"statusCode": 200, "body": json.dumps(results)}
