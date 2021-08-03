from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pathlib import Path
from . import util as DataUtil


FOLDER_MIME = "application/vnd.google-apps.folder"
PARENT_KIND = "drive#fileLink"


class Drive:
    def __init__(self, creds_path="drive_creds.json"):
        self.client = self.auth(creds_path=creds_path)

    def auth(self, creds_path="drive_creds.json"):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(creds_path)
        if gauth.credentials is None:
            gauth.credentials = GoogleCredentials.get_application_default()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile(creds_path)
        client = GoogleDrive(gauth)
        self.client = client
        return client

    def download_file(self, file_id, save_path):
        DataUtil.mkparent(Path(save_path))
        drive_file = self.client.CreateFile({
            "id": file_id
        })
        drive_file.GetContentFile(save_path)
        return drive_file

    def upload_file(self, file_path, parent_id, file_name=None):
        file_path_2 = Path(file_path)
        DataUtil.mkparent(file_path_2)
        file_name = file_name or file_path_2.name
        drive_file = self.client.CreateFile({
            "title": file_name, 
            "parents": [{
                "id": parent_id,
                "kind": PARENT_KIND,
            }]
        })
        drive_file.SetContentFile(file_path)
        drive_file.Upload()
        return drive_file

    def update_file(self, file_path, file_id, file_name=None):
        file_path_2 = Path(file_path)
        DataUtil.mkparent(file_path_2)
        file_name = file_name or file_path_2.name
        drive_file = self.client.CreateFile({
            "title": file_name,
            "id": file_id
        })
        drive_file.SetContentFile(file_path)
        drive_file.Upload()
        return drive_file

    def create_folder(self, folder_name, parent_id):

        folders = self.client.ListFile(
            {'q': f"title='{folder_name}' and '{parent_id}' in parents and trashed=false".format()}
        ).GetList()
        if len(folders) > 0:
            return folders[0]

        folder = self.client.CreateFile({
            "title": folder_name,
            "mimeType": FOLDER_MIME,
            "parents": [{
                "kind": PARENT_KIND, 
                "id": parent_id
            }]
        })
        folder.Upload()

        return folder

    def get_folder_files(self, folder_id):
        files = self.client.ListFile(
            {'q': "'{0}' in parents and trashed=false".format(folder_id)}
        ).GetList()
        return files

    def find_files(self, file_names, parent_id=None, files=None):
        assert parent_id is not None or files is not None
        if files is None:
            files = self.get_folder_files(parent_id)
        files = {
            **{file_name: None for file_name in file_names},
            **{file['title']: file for file in files if file['title'] in file_names}
        }
        return files


    def upload_folder(self, folder_path, parent_id, only_contents=False, update=True):
        folder = Path(folder_path)
        folders, files = DataUtil.get_sub_folders_files(folder)

        if not only_contents:
            parent_id = self.create_folder(folder.name, parent_id)["id"]

        existing = self.get_folder_files(parent_id)

        existing_files = self.find_files(
            file_names=[str(file) for file in files],
            files=existing
        )
        for file in files:
            file_path = f"{folder_path}/{str(folder)}"
            if update and file.name in existing_files:
                self.update_file(
                    file_path,
                    file_id=existing_files[file.name]["id"]
                )
            else:
                self.upload_file(
                    file_path,
                    parent_id=parent_id
                )

        existing_folders = self.find_files(
            file_names=[str(folder) for folder in folders],
            files=existing
        )
        for folder in folders:
            if update and folder in existing_folders:
                self.upload_folder(
                    f"{folder_path}/{str(folder)}",
                    parent_id=existing_folders[folder]["id"],
                    only_contents=True
                )
            else:
                self.upload_folder(
                    f"{folder_path}/{str(folder)}",
                    parent_id=parent_id,
                    only_contents=False
                )
        drive_parent = self.client.CreateFile({
            "id": parent_id
        })
        return drive_parent

