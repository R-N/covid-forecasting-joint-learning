from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pathlib import Path
from . import util as DataUtil


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
			'id': file_id
		})
		drive_file.GetContentFile(save_path)
		return drive_file

	def upload_file(self, file_path, parent_id, file_name=None):
		file_path_2 = Path(file_path)
		DataUtil.mkparent(file_path_2)
		file_name = file_name or file_path_2.name
		drive_file = self.client.CreateFile({
			'title': file_name, 
			'parents': [{
				'id': parent_id,
				"kind": "drive#fileLink",
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
			'title': file_name,
			'id': file_id
		})
		drive_file.SetContentFile(file_path)
		drive_file.Upload()
		return drive_file

