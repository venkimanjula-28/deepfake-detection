import sqlite3
import bcrypt
import os
from datetime import datetime
import streamlit as st

class UserDatabase:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Create user_sessions table for tracking login sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Create analysis_history table to track user analyses
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    file_name TEXT,
                    file_type TEXT,
                    prediction TEXT,
                    confidence REAL,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            conn.commit()

    def hash_password(self, password):
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def verify_password(self, password, hashed):
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    def create_user(self, username, password, email=None, full_name=None):
        """Create a new user account"""
        try:
            password_hash = self.hash_password(password)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, password_hash, email, full_name)
                    VALUES (?, ?, ?, ?)
                ''', (username, password_hash, email, full_name))
                conn.commit()
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("Username already exists")

    def authenticate_user(self, username, password):
        """Authenticate a user and return user info if successful"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash, email, full_name, is_active
                FROM users
                WHERE username = ? AND is_active = 1
            ''', (username,))

            user = cursor.fetchone()
            if user and self.verify_password(password, user[2]):
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user[0],))
                conn.commit()

                return {
                    'id': user[0],
                    'username': user[1],
                    'email': user[3],
                    'full_name': user[4],
                    'is_active': user[5]
                }
        return None

    def get_user_by_id(self, user_id):
        """Get user information by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, email, full_name, created_at, last_login, is_active
                FROM users
                WHERE id = ?
            ''', (user_id,))

            user = cursor.fetchone()
            if user:
                return {
                    'id': user[0],
                    'username': user[1],
                    'email': user[2],
                    'full_name': user[3],
                    'created_at': user[4],
                    'last_login': user[5],
                    'is_active': user[6]
                }
        return None

    def log_analysis(self, user_id, file_name, file_type, prediction, confidence):
        """Log an analysis performed by a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_history (user_id, file_name, file_type, prediction, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, file_name, file_type, prediction, confidence))
            conn.commit()

    def get_user_analysis_history(self, user_id, limit=10):
        """Get user's analysis history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_name, file_type, prediction, confidence, analyzed_at
                FROM analysis_history
                WHERE user_id = ?
                ORDER BY analyzed_at DESC
                LIMIT ?
            ''', (user_id, limit))

            return cursor.fetchall()

    def get_user_stats(self, user_id):
        """Get user statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total analyses
            cursor.execute('SELECT COUNT(*) FROM analysis_history WHERE user_id = ?', (user_id,))
            total_analyses = cursor.fetchone()[0]

            # Fake vs Real detections
            cursor.execute('''
                SELECT prediction, COUNT(*) as count
                FROM analysis_history
                WHERE user_id = ?
                GROUP BY prediction
            ''', (user_id,))

            detection_stats = dict(cursor.fetchall())

            return {
                'total_analyses': total_analyses,
                'fake_detections': detection_stats.get('FAKE', 0),
                'real_detections': detection_stats.get('REAL', 0)
            }


# Global database instance
db = UserDatabase()


def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'login'


def require_auth():
    """Decorator to require authentication for a page"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not st.session_state.get('authenticated', False):
                show_login_page()
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator


def show_login_page():
    """Display the login/signup page"""
    st.markdown("""
    <style>
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        .auth-header {
            text-align: center;
            margin-bottom: 2rem;
            color: #1f77b4;
        }
        .auth-tabs {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .auth-tab {
            padding: 0.5rem 1rem;
            border: none;
            background: none;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }
        .auth-tab.active {
            border-bottom-color: #1f77b4;
            color: #1f77b4;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="auth-header">🔐 DeepFake Detector Login</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

    with tab1:
        st.subheader("Login to Your Account")

        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                if not username or not password:
                    st.error("Please fill in all fields")
                else:
                    user = db.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.current_page = 'main'
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

    with tab2:
        st.subheader("Create New Account")

        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username")
            new_email = st.text_input("Email (optional)", key="signup_email")
            new_full_name = st.text_input("Full Name (optional)", key="signup_full_name")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            signup_button = st.form_submit_button("Create Account", use_container_width=True)

            if signup_button:
                if not new_username or not new_password:
                    st.error("Username and password are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    try:
                        user_id = db.create_user(new_username, new_password, new_email, new_full_name)
                        st.success("Account created successfully! Please login.")
                    except ValueError as e:
                        st.error(str(e))


def show_user_profile():
    """Display user profile and statistics"""
    if not st.session_state.get('authenticated', False):
        return

    user = st.session_state.user
    stats = db.get_user_stats(user['id'])

    with st.sidebar:
        st.markdown("---")
        st.subheader("👤 User Profile")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("👤")
        with col2:
            st.write(f"**{user['username']}**")
            if user['full_name']:
                st.write(user['full_name'])
            if user['email']:
                st.write(user['email'])

        st.markdown("### 📊 Your Statistics")
        st.metric("Total Analyses", stats['total_analyses'])
        st.metric("Fake Detections", stats['fake_detections'])
        st.metric("Real Detections", stats['real_detections'])

        if st.button("📋 View History"):
            st.session_state.current_page = 'history'

        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.current_page = 'login'
            st.rerun()


def show_analysis_history():
    """Display user's analysis history"""
    if not st.session_state.get('authenticated', False):
        return

    user = st.session_state.user
    history = db.get_user_analysis_history(user['id'], limit=50)

    st.header("📋 Analysis History")

    if not history:
        st.info("No analysis history yet. Start by analyzing some media files!")
        return

    # Create a dataframe for better display
    import pandas as pd

    df = pd.DataFrame(history, columns=['File Name', 'Type', 'Prediction', 'Confidence', 'Date'])
    df['Confidence'] = df['Confidence'].apply(lambda x: f"{x:.1%}")
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')

    st.dataframe(df, use_container_width=True)

    # Summary statistics
    fake_count = len(df[df['Prediction'] == 'FAKE'])
    real_count = len(df[df['Prediction'] == 'REAL'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyses", len(df))
    with col2:
        st.metric("Fake Detected", fake_count)
    with col3:
        st.metric("Real Detected", real_count)

    if st.button("⬅️ Back to Main"):
        st.session_state.current_page = 'main'
        st.rerun()