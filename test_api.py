from google_auth_oauthlib.flow import InstalledAppFlow

flow = InstalledAppFlow.from_client_secrets_file(
    'client_secret.json',
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)

creds = flow.run_local_server(port=8080)
print("Access token:", creds.token)
