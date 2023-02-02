import os.path
from flask import Flask, render_template, request, redirect, Response
from werkzeug.utils import secure_filename

from hubconfCustom import video_detection
import hubconfCustom
import cv2

app = Flask(__name__, static_folder = 'static')
app.config["VIDEO_UPLOADS"] = "static/video"
app.config["ALLOWED_VIDEO_EXTENSIONS"] = ["MP4", "MOV", "AVI", "WMV"]

frames_buffer = []
vid_path = 'static/video/vid.mp4'

def allowed_video(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return True
    else:
        return False


def generate_raw_frames(path_x =''):
    '''A function to yield frames from vid.mp4'''
    video = cv2.VideoCapture(path_x)
    while True:
        success, frame = video.read()
        
        # if path_x starts with http, this is a stream
        if path_x.startswith('http://'):
            #stack it on the buffer
            frames_buffer.append(frame)

        if not success:
            pass
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


def generate_processed_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_, frames_buffer)
    for detection_,FPS_,xl,yl in yolo_output:
	    #The function imencode compresses the image and stores it in the memory buffer that is resized to fit the result.
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video_processed')
def video_processed():
    return Response(generate_processed_frames(path_x = vid_path,conf_=0.75),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_raw')
def video_raw():
    return Response(generate_raw_frames(path_x = vid_path),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET", "POST"])
def index():
    global vid_path
    if request.method == "POST":
        if 'video_upload_button' in request.form:
            print('Upload Video Button Clicked')
            video = request.files['video']

            if video.filename == "":
                print("Video must have a filename")

            '''if not allowed_video(video.filename):
                print("That video extension is not allowed")
                return redirect(request.url)
            else:
                filename = secure_filename(video.filename)'''
            
            filename = 'vid.mp4'
            video.save(os.path.join(app.config["VIDEO_UPLOADS"], filename))
            print("Video saved")
            return redirect(request.url)
        
        elif 'inference_video_button' in request.form:
            print('Inference Video Button Clicked')
            vid_path = 'static/video/vid.mp4'
            return render_template('index.html')
        
        elif 'live_inference_button' in request.form:
            print('Live Inference Button Clicked')
            vid_path = 'http://192.168.12.10:4747/video'
            print('Life inference running on : ',vid_path)
            return render_template('index.html')
        
        elif 'download_button' in request.form:
            print('Download Button Clicked')
            print(f'Detections Summary: {hubconfCustom.detections_summary}')
            #save detections_summary as a text file
            with open('static/reports/detections_summary.txt', 'w') as f:
                f.write(hubconfCustom.detections_summary)
            return Response(open('static/reports/detections_summary.txt', 'rb').read(),
                        mimetype='text/plain',
                        headers={"Content-Disposition":"attachment;filename=detections_summary.txt"})
            #summary = detection_summary(vid_path)
            #return render_template('index.html', summary = summary)

    
    return render_template('index.html')


if __name__ == "__main__":
    #copy file from static/files/vid.mp4 to static/video/vid.mp4
    os.system('cp static/files/vid.mp4 static/video/vid.mp4')
    app.run(debug=True)
