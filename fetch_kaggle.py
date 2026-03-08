import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. The EXACT path you want
target_path = r"C:\Users\UTKARSH\.cache\kagglehub\datasets\iarunava\cell-images-for-detecting-malaria\versions\1"

# 2. Ensure folder exists
os.makedirs(target_path, exist_ok=True)

try:
    api = KaggleApi()
    api.authenticate()
    print("Kaggle Authenticated Successfully.")

    # 3. Download the ZIP file (we won't use 'unzip=True' this time)
    print("Downloading dataset... this will take a minute.")
    api.dataset_download_files("iarunava/cell-images-for-detecting-malaria", path=target_path)

    # 4. Manually Unzip
    zip_name = "cell-images-for-detecting-malaria.zip"
    zip_path = os.path.join(target_path, zip_name)

    if os.path.exists(zip_path):
        print(f"File downloaded. Extracting {zip_name} now...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)
        print("Extraction complete!")
        
        # Clean up the zip file to save space
        os.remove(zip_path)
        print(f"SUCCESS: Your files are ready at: {target_path}")
    else:
        print("ERROR: The zip file was not found. Check your internet connection.")

except Exception as e:
    print(f"An error occurred: {str(e)}")