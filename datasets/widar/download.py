import io
import os
import zipfile

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define the Google Drive folder ID
# Manually download from https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt
FOLDER_ID = "1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt"

# Define the directory where you want to save the dataset
SAVE_DIR = "./"

# Add your service account file path
SERVICE_ACCOUNT_FILE = "path/to/your-service-account-file.json"


# Function to create Google Drive API client
def create_google_drive_client(service_account_file):
    credentials = service_account.Credentials.from_service_account_file(service_account_file)
    try:
        service = build("drive", "v3", credentials=credentials)
        return service
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


# Function to download the file from Google Drive
def download_file_from_google_drive(service, file_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    request = service.files().get_media(fileId=file_id)
    file = io.BytesIO(request.execute())
    zip_ref = zipfile.ZipFile(file, "r")
    zip_ref.extractall(save_dir)
    print(f"Extracted dataset to {save_dir}")


# Function to list all files in the specified Google Drive folder
def list_files_in_google_drive_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get("files", [])

    return items


# Main function to download and extract the WidarData.zip file
def main():
    service = create_google_drive_client(SERVICE_ACCOUNT_FILE)
    if service is not None:
        files = list_files_in_google_drive_folder(service, FOLDER_ID)
        for file in files:
            if file["name"] == "WidarData.zip":
                download_file_from_google_drive(service, file["id"], SAVE_DIR)
                break
        else:
            print("WidarData.zip not found in the specified folder.")


if __name__ == "__main__":
    main()
