import cv2
from flask import Flask, Response, jsonify, render_template, request, session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from hubconfCustom import video_detection

app = Flask(__name__, static_folder = 'templates/assets/')
Bootstrap(app)

#The secret key helps to maintain a user session
app.config['SECRET_KEY'] = 'grilsessionkey'

@app.route("/",methods=['GET','POST'])
@app.route("/home", methods=['GET','POST'])
def home():
	session.clear()
	return render_template('root.html')


def generate_frames(path_x = '',conf_= 0.25):
    yolo_output = video_detection(path_x,conf_)
    for detection_,FPS_,xl,yl in yolo_output:
	    #The function imencode compresses the image and stores it in the memory buffer that is resized to fit the result.
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
            
                    
@app.route('/FrontPage')
@app.route('/video')
def video():
    return Response(generate_frames(path_x = 'static/files/vid.mp4',conf_=0.75),mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == "__main__":
    app.run(debug=True)
