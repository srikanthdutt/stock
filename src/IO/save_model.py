# https://cloud.google.com/storage/docs/reference/libraries
# https://googleapis.dev/python/storage/latest/client.html

import logging
import joblib

from keras.models import load_model
from google.api_core.exceptions import NotFound
from google.cloud import storage


def upload_file_to_bucket(model_file_name, bucket_name):
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(model_file_name, b)
    with open(model_file_name, "rb") as model_file:
        blob.upload_from_file(model_file)


def get_model_from_gcp_bucket(model_filename, bucket_name):
    client = storage.Client()
    b = client.get_bucket(bucket_name)
    blob = storage.Blob(f'{model_filename}', b)
    try:
        with open(f'{model_filename}', 'wb') as file_obj:  # critical resource should use tempfile...
            client.download_blob_to_file(blob, file_obj)

        if 'h5' in model_filename:
            model = load_model(model_filename)
        else:
            with open(f'{model_filename}', 'rb') as file_obj:
                model = joblib.load(file_obj)
    except NotFound as e:
        model = None

    return model
