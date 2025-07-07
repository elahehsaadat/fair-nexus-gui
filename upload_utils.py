import paramiko
from urllib.parse import urlparse
import requests

def send_file_sftp(local_path, remote_uri):
    parsed = urlparse(remote_uri)
    transport = paramiko.Transport((parsed.hostname, 22))
    transport.connect(username=parsed.username)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(local_path, parsed.path)
    sftp.close()
    transport.close()
    print(f"📤 File uploaded to {remote_uri}")

def send_file_http(local_path, url):
    with open(local_path, 'rb') as f:
        res = requests.post(url, files={'file': f})
    print(f"📤 Uploaded to {url}, status code: {res.status_code}")
