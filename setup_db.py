#!/usr/bin/env python3
"""
Setup script to initialize the database and create initial users.
"""

from src.database import db

def setup_database():
    """Initialize database and create sample users"""
    print("🔧 Setting up DeepFake Detector Database...")

    # Create sample users
    sample_users = [
        {
            'username': 'admin',
            'password': 'admin123',
            'email': 'admin@deepfake.com',
            'full_name': 'System Administrator'
        },
        {
            'username': 'demo',
            'password': 'demo123',
            'email': 'demo@deepfake.com',
            'full_name': 'Demo User'
        }
    ]

    for user_data in sample_users:
        try:
            user_id = db.create_user(
                user_data['username'],
                user_data['password'],
                user_data['email'],
                user_data['full_name']
            )
            print(f"✅ Created user: {user_data['username']} (ID: {user_id})")
        except ValueError as e:
            print(f"⚠️  User {user_data['username']} already exists")

    print("\n🎉 Database setup complete!")
    print("\n📋 Sample Login Credentials:")
    print("   Admin: admin / admin123")
    print("   Demo:  demo  / demo123")
    print("\n🚀 You can now run: streamlit run app_streamlit.py")

if __name__ == '__main__':
    setup_database()