import os
import sys
import time
import django
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.core.management import call_command
from django.db import connection

def reset_database():
    print("Starting database reset process...")
    
    # Get the absolute path to the database file
    db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
    print(f"Database path: {db_path}")
    
    # Close any existing connections
    connection.close()
    print("Closed existing database connections")
    
    # Wait a moment
    time.sleep(2)
    
    # Try to remove the database file
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print("✓ Successfully removed old database file")
        else:
            print("No existing database file found")
    except Exception as e:
        print(f"Error removing database file: {e}")
        return False
    
    # Run migrations to create new database
    try:
        print("\nCreating new database...")
        call_command('migrate')
        print("✓ Successfully created new database")
        return True
    except Exception as e:
        print(f"Error creating new database: {e}")
        return False

if __name__ == '__main__':
    print("=== Database Reset Tool ===")
    success = reset_database()
    if success:
        print("\n✓ Database reset completed successfully!")
    else:
        print("\n✗ Database reset failed!")
    input("\nPress Enter to exit...") 