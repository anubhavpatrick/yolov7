'''
This module sends emails with attachments to the participants
Reference - https://developers.google.com/gmail/api/quickstart/python
'''

from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import mimetypes
import os
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
import base64
import cv2


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def aunthentication():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def prepare_and_send_email(recipient, subject, message_text, im0):
    """Prepares and send email with attachment to the participants 
    """
    creds = aunthentication()

    try:
        # Call the Gmail API
        service = build('gmail', 'v1', credentials=creds)

        #create message 
        msg = create_message('anubhav.patrick@giindia.com', recipient, subject, message_text, im0)
        send_message(service, 'me', msg)

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f'An error occurred: {error}')

    

def create_message(sender, to, subject, message_text, img_file):
    """Create a message for an email.

    Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    img_file: The image to be attached

    Returns:
    An object containing a base64url encoded email object.
    """
    message = MIMEMultipart()

    message['from'] = sender
    message['to'] = to
    message['subject'] = subject

    location = 'ABESIT'
    #get current date and time
    current_date_time = time.time()
    formatted_date_time = time.strftime("%H:%M:%S_%d-%m-%Y", time.localtime(current_date_time))
    file_name = 'violation_'+location + '_' + formatted_date_time + '.jpg'

    #convert img_file into jpeg
    cv2.imencode('.jpg', img_file)[1].tofile(file_name)

    msg = MIMEText(message_text)
    message.attach(msg)

    content_type, encoding = mimetypes.guess_type(file_name)
    main_type, sub_type = content_type.split('/', 1)

    if main_type == 'text':
        fp = open(file_name, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(file_name, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'application' and sub_type=='pdf' and encoding == None: #Reference - https://coderzcolumn.com/tutorials/python/mimetypes-guide-to-determine-mime-type-of-file
        print("INSIDE PDF")
        main_type = 'application'
        sub_type = 'octet-stream'
        fp = open(file_name, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        encoders.encode_base64(msg)
        fp.close()
    else:
        fp = open(file_name, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    print(main_type,sub_type,encoding)
    filename = os.path.basename(file_name)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def send_message(service, user_id, message):
    """Send an email message.

    Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

    Returns:
    Sent Message.
    """
    try:
        message = (service.users().messages().send(userId=user_id, body=message)
                    .execute())
        print('Message Id: %s' % message['id'])
        return message
    except HttpError as error:
        print('An error occurred: %s' % error)


if __name__ == '__main__':
    #prepare_and_send_email('anubhavpatrick@gmail.com', 'Greeting from Global Infoventures', 'This is a test email for our upcoming app')
    pass