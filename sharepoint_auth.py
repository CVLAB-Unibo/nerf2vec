import os
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.creation_information import FileCreationInformation
import getpass

def print_upload_progress(offset, local_path):
    file_size = os.path.getsize(local_path)
    print("Uploaded '{0}' bytes from '{1}'...[{2}%]".format(offset, file_size, round(offset / file_size * 100, 2)))

def upload_file_to_sharepoint_chunked(client_context, local_path, folder_url, file_name, size_chunk):
    with open(local_path, 'rb') as content_file:
        target_folder = client_context.web.get_folder_by_server_relative_url(folder_url)
        uploaded_file = target_folder.files.create_upload_session(
            content_file, size_chunk,
            lambda offset: print_upload_progress(offset, local_path)
        ).execute_query()

        # print('File {0} has been uploaded successfully'.format(uploaded_file.serverRelativeUrl))
    
    print(f"File {file_name} uploaded successfully.")

"""
def upload_file_to_sharepoint(client_context, local_path, folder_url, file_name):
    with open(local_path, 'rb') as content_file:
        file_content = content_file.read()

    info = FileCreationInformation()
    info.content = file_contentF
    info.url = file_name
    info.overwrite = True

    target_folder = client_context.web.get_folder_by_server_relative_url(folder_url)
    target_folder.upload_file(info.url, file_content)
    client_context.execute_query()
    print(f"File {file_name} uploaded successfully.")
"""

site_url = "https://liveunibo.sharepoint.com/sites/CVLab"
username = "daniele.sirocchi@studio.unibo.it"
password = getpass.getpass("Enter your password: ")
folder_path = "data_backup/task_mapping_network/"  # Local
relative_url = "/sites/CVLab/Shared Documents/Datasets/nerf2vec/task_mapping_network"
size_chunk = 200000000
already_uploaded = [
]

ctx_auth = AuthenticationContext(site_url)
if ctx_auth.acquire_token_for_user(username, password):
    ctx = ClientContext(site_url, ctx_auth)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            print(f'uploading {file_path}...')
            # upload_file_to_sharepoint(ctx, file_path, relative_url, file_name)
            # local_path = file_path
            if file_path in already_uploaded:
                print(f'ALREADY UPLOADED...')
                continue
            upload_file_to_sharepoint_chunked(ctx, file_path, relative_url, file_name, size_chunk)
            
else:
    print(ctx_auth.get_last_error())
