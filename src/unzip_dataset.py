import zipfile
import os

def unzip_plant_village(zip_path, extract_path):
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found!")
        return
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Extracting {zip_path}...")
        zip_ref.extractall(extract_path)
        print("Extraction complete!")

if __name__ == "__main__":
    zip_path = "data/PlantVillage.zip"  # Path to the ZIP file
    extract_path = "data/"  # Extract into data folder

    unzip_plant_village(zip_path, extract_path)
