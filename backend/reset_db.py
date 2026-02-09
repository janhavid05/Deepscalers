import os
import django
import time

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.core.management import call_command
from django.db import connection

def reset_database():
    print("Starting database reset...")
    
    # Close any existing connections
    connection.close()
    
    # Wait a moment to ensure the database is not in use
    time.sleep(1)
    
    # Get the database file path
    db_path = 'db.sqlite3'
    
    # Try to remove the database file
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print("✓ Old database file removed")
        else:
            print("No existing database file found")
    except Exception as e:
        print(f"Error removing database file: {e}")
        return
    
    # Run migrations to create new database
    try:
        print("Creating new database...")
        call_command('migrate')
        print("✓ Database reset complete!")
    except Exception as e:
        print(f"Error creating new database: {e}")

if __name__ == '__main__':
    reset_database() 