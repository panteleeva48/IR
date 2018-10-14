from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Done!</h1>'
    
if name == '__name__':
    app.run()