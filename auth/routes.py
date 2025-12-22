from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from dotenv import load_dotenv
import os

from models.user import create_user, get_user_by_email, verify_user_email

# Load .env inside auth
load_dotenv()

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# Extensions
bcrypt = Bcrypt(app)
mail = Mail(app)
CORS(app)

# ---------------- Routes ----------------

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/sign_up')
def sign_up_page():
    return render_template('sign_up.html')

@app.route('/login_page')
def login_page():
    return render_template('login.html')


# Signup Route
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data['email']
    password = data['password']

    if not (email.endswith('@gmail.com') or email.endswith('@iittnif.com')):
        return jsonify({'error': 'Only @gmail.com and @iittnif.com domains allowed'}), 400

    if get_user_by_email(email):
        return jsonify({'error': 'User already exists'}), 400

    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    create_user(email, pw_hash)

    s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
    token = s.dumps(email, salt='email-verify')
    
    # Updated link with port 5002 as requested
    link = f"http://localhost:5002/verify/{token}"

    msg = Message('Verify your email',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[email])
    msg.body = f'Click to verify: {link}'
    mail.send(msg)

    return jsonify({'message': 'Verification link sent to email'})


# Email Verification Route
@app.route('/verify/<token>')
def verify_email(token):
    try:
        s = URLSafeTimedSerializer(app.config['SECRET_KEY'])
        email = s.loads(token, salt='email-verify', max_age=3600)
        verify_user_email(email)
        return "Email verified successfully! You can now log in."
    except Exception:
        return "Invalid or expired token", 400


# Login Route
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']

    user = get_user_by_email(email)
    if not user:
        return jsonify({'error': 'User not found'}), 401

    # Ensure DB returns correct tuple format
    user_id, user_email, user_password, reset_token, is_verified = user

    if not is_verified:
        return jsonify({'error': 'Email not verified'}), 403

    if not bcrypt.check_password_hash(user_password, password):
        return jsonify({'error': 'Invalid password'}), 401
    session["user_id"] = user_id 

    return jsonify({'success': True, 'redirect': '/admin2', 'message': 'Login successful'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
