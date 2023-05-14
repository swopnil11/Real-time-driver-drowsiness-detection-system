from flask import Flask, render_template, url_for, redirect, Response, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
from pygame import mixer

app = Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError("The username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Enter your Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Enter your Password"})

    submit = SubmitField('Login')


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
        flash("Invalid Username or password!", "danger")
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    form = LoginForm()
    username=form.username.data
    return render_template('dashboard.html', username=username)

@app.route('/main', methods=['GET', 'POST'])
@login_required
def main():
    form = LoginForm()
    username=form.username.data
    return render_template('main.html', username=username)

def gen_frames():
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    model = load_model(r'E:\Major Projects\Driver-Drowsiness-Detection-using-Deep-Learning-main\models\modeltwo.h5')
    mixer.init()
    sound= mixer.Sound(r'E:\Major Projects\Driver-Drowsiness-Detection-using-Deep-Learning-main\alarm.wav')
    Score = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            #ret, frame = camera.read()
            height,width = frame.shape[0:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
            eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
    
            cv2.rectangle(frame,(180,0),(400,40),(0,0,0),thickness=cv2.FILLED)
    
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
        
            for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color= (255,0,0), thickness=3 )
        
            # preprocessing steps
                eye= frame[ey:ey+eh,ex:ex+ew]
                eye= cv2.resize(eye,(80,80))
                eye= eye/255
                eye= eye.reshape(80,80,3)
                eye= np.expand_dims(eye,axis=0)
                # preprocessing is done now model prediction
                prediction = model.predict(eye)
        
                # if eyes are closed
                if prediction[0][0]>0.30:
                    cv2.putText(frame,'closed',(200,height-450),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                            thickness=1,lineType=cv2.LINE_AA)
                    cv2.putText(frame,'Score'+str(Score),(300,height-450),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                            thickness=1,lineType=cv2.LINE_AA)
                    Score=Score+1
                    if(Score>10):
                        try:
                            sound.play()
                        except:
                            pass
                    
                # if eyes are open
                elif prediction[0][1]>0.90:
                    cv2.putText(frame,'open',(200,height-450),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                            thickness=1,lineType=cv2.LINE_AA)      
                    cv2.putText(frame,'Score'+str(Score),(300,height-450),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),
                            thickness=1,lineType=cv2.LINE_AA)
                    Score = 0
                    if (Score<0):
                        Score=0

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
