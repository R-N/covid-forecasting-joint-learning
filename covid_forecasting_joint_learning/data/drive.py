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
			{'q': "'{0}' in parents and trashed=false".format(parent_id)}
		).GetList()
        return files

    def find_files(self, file_names, folder_id=None, files=None):
    	assert folder_id is not None or files is not None
    	if files is None:
    		files = self.get_folder_files(folder_id)
    	files = {
    		**{file_name: None for file_name in file_names},
    		**{file['title']: file for file in files if file['title'] in file_names}
		}
		return files

