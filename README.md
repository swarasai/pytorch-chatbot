Install python 3.11.9 torch 2.4.1+cpu

Run python script 'python flask_app.py' in terminal, this will start the flask application

Need to install below 
pip install flask 
pip install flask_cors 
pip install sentence_transformers 
pip install fuzzywuzzy 
pip install pyspellchecker

Run web app in another terminal using 'python -m http.server 8000' this will start the School website, which uses the above flask app which is using the 'all-MiniLM-L6-v2' model to implement a chatbot.