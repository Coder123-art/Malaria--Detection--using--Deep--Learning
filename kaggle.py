try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed. Install it using: pip install kaggle")
        exit(1)

# Authenticate and download dataset
api = KaggleApi()
api.authenticate()
api.dataset_download_files("iarunava/cell-images-for-detecting-malaria", path=".", unzip=True)

print("Dataset downloaded successfully")