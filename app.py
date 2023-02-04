'''A Flask application to run the YOLOv7 PPE violation model on a video file or ip cam stream

Authors: Anubhav Patrick and Hamza Aziz
Date: 3/02/2023 
'''

from flask import flash
import os.path
from flask import Flask, render_template, request, redirect, Response

from hubconfCustom import video_detection
import hubconfCustom
import cv2

# Initialize the Flask application
app = Flask(__name__, static_folder = 'static')
app.config["VIDEO_UPLOADS"] = "static/video"
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4", "MOV", "AVI", "WMV"]

#secret key for the session
app.config['SECRET_KEY'] = 'ppe_violation_detection'

#global variables
frames_buffer = [] #buffer to store frames from a stream
vid_path = 'static/video/vid.mp4' #path to uploaded/stored video file 
can_send_email = False #flag to ensure whether or whether not send email
email_recipient = 'anubhav.patrick@giindia.com' #default email address of the recipient


def allowed_video(filename: str):
    '''A function to check if the uploaded file is a video
    
    Args:
        filename (str): name of the uploaded file

    Returns:
        bool: True if the file is a video, False otherwise
    '''
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False


def generate_raw_frames(path_x):
    '''A function to yield unprocessed frames from stored video file or ip cam stream
    
    Args:
        path_x (str, optional): path to the video file or ip cam stream.
    
    Yields:
        bytes: a frame from the video file or ip cam stream
    '''
    # capture video from file or ip cam stream
    video = cv2.VideoCapture(path_x)
    
    while True:
        # Keep reading the frames from the video file or ip cam stream
        success, frame = video.read()

        if not success:
            #if there are no more frames to read, break the loop
            # Potential bug: if prog is not able to connect to ip cam stream immidiately, this will break the code
            #break
            pass   
        else:
            # if path_x starts with http, this is an ip stream
            # We will read it once and store it in the buffer for the inference function
            if path_x.startswith('http://'):
                #stack it on the buffer to be used by the inference function
                frames_buffer.append(frame) 
            
            #compress the frame and store it in the memory buffer
            _, buffer = cv2.imencode('.jpg', frame) 
            #convert the buffer to bytes
            frame = buffer.tobytes() 
            #yield the frame to the browser
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n') 


def generate_processed_frames(path_x,conf_= 0.25):
    '''A function to yield processed frames from stored video file or ip cam stream after violation detection
    
    Args:
        path_x (str): path to the video file or ip cam stream.
        conf_ (float, optional): confidence threshold for the detection. Defaults to 0.25.
    
    Yields:
        bytes: a processed frame from the video file or ip cam stream
    '''
    #call the video_detection for violation detection which yields a list of processed frames
    yolo_output = video_detection(path_x, conf_, frames_buffer, can_send_email, email_recipient)
    #iterate through the list of processed frames
    for detection_, _, _, _ in yolo_output:
	    #The function imencode compresses the image and stores it in the memory buffer 
        _,buffer=cv2.imencode('.jpg',detection_)
        #convert the buffer to bytes
        frame=buffer.tobytes()
        #yield the processed frame to the browser
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video_raw')
def video_raw():
    return Response(generate_raw_frames(path_x = vid_path),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_processed')
def video_processed():
    return Response(generate_processed_frames(path_x = vid_path,conf_=0.75),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET", "POST"])
def index():
    '''A function to handle the requests from the web page

    Args:
        None
    
    Returns:
        render_template: the index.html page
    '''
    global vid_path, can_send_email, email_recipient

    #if the request is a POST request made by user interaction with the HTML form
    if request.method == "POST":

        if 'video_upload_button' in request.form:
            #print('Upload Video Button Clicked')
            video = request.files['video']

            if video.filename == "":
                print("Video must have a filename")
                # display a flash alert message on the web page
                flash("That video must have a file name", "error")

            elif not allowed_video(video.filename):
                print("That video extension is not allowed")
                # display a flash alert message on the web page
                flash("That video extension is not allowed", "error")
                # return redirect(request.url)
            
            else:
                filename = 'vid.mp4'
                video.save(os.path.join(app.config["VIDEO_UPLOADS"], filename))
                print("Video saved")
                # display a flash alert message on the web page
                flash("That video extension is successfully uploaded", "success")
                #return redirect(request.url)
        
        elif 'inference_video_button' in request.form:
            print('Inference Video Button Clicked')
            # reset vid_path to the default video
            vid_path = 'static/video/vid.mp4'
            return render_template('index.html')
        
        elif 'live_inference_button' in request.form:
            print('Live Inference Button Clicked')
            #read text box value
            vid_ip_path = request.form['ip_address_textbox']
            #check if vid_ip_path is a valid url
            if vid_ip_path.startswith('http://'):
                vid_path = vid_ip_path.strip()
            else:
                # the url is not valid
                flash("Incorrect URL for ip cam", "error")
                # re-render the index.html page
                return render_template('index.html')
            
            print('Life inference running on : ',vid_path)
            return render_template('index.html')
        
        elif 'download_button' in request.form:
            print('Download Button Clicked')
            with open('static/reports/detections_summary.txt', 'w') as f:
                f.write(hubconfCustom.detections_summary)
            return Response(open('static/reports/detections_summary.txt', 'rb').read(),
                        mimetype='text/plain',
                        headers={"Content-Disposition":"attachment;filename=detections_summary.txt"})
        
        elif 'alert_email_checkbox' in request.form:
            print('Alert Email Checkbox Checked')
            email_recipient = request.form['alert_email_textbox']
            if can_send_email:
                can_send_email = False
                # display a flash alert message on the web page
                flash("Alert email is disabled", "success")    
            else:
                can_send_email = True
                # display a flash alert message on the web page
                flash(f"Alert email is enabled at {email_recipient}", "success")

    return render_template('index.html')

if __name__ == "__main__":
    #copy file from static/files/vid.mp4 to static/video/vid.mp4
    os.system('cp static/files/vid.mp4 static/video/vid.mp4')
    app.run(debug=True)
