from crypt import methods
import os
import flask
import requests
import json

from flask import render_template, redirect, url_for, request
from flask_socketio import SocketIO, send, emit
from flask_bootstrap import Bootstrap
from App.translationService import chat, chat_text

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'hackmit2021'
socketio = SocketIO(app)

@app.route('/')
def index():
    print('34556')
    return render_template('index.html')

@socketio.on('message')
def handleMessage(msg):
    if(msg is None or len(msg) == 0):
        return
    print('Message: ' + msg)
    textbased = chat_text(msg)

    send(textbased, broadcast=True)

@socketio.on('my event')
def handle_my_custom_event(data):
    print('received my event: ' + str(data))
   
    chat(socketio)


Bootstrap(app)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=1123)
