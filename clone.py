from flask import Flask
import os
app = Flask(__name__,template_folder='templates',static_folder='static',static_url_path='/static')
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
for i in [app.config['UPLOAD_FOLDER'],app.config['PROCESSED_FOLDER']]:
    if not os.path.exists(i):
        os.makedirs(i)
processing_jobs = {}

DEFAULT_SETTINGS = {}

COLOR_PALETTES = {


}





