# ipatient_deploy

USE PYTHON 3.7 ONLY

Use these commands to install the dependencies:

  pip3 install sentence-transformers 
  
  pip3 install textblob 
  
  sudo pip3 install --upgrade python-socketio==4.6.0 
  
  sudo pip3 install --upgrade python-engineio==3.13.2 
  
  sudo pip3 install --upgrade Flask-SocketIO==4.3.1 
  

sudo ./run.sh to start

Open a web browser on 127.0.0.1:80 and press the record button to talk to the virtual patient.

Use the keyword "exit" or say "exit" to end the session.

Once exited the responses and the empathy measurement is saved in the root directory as output.csv
