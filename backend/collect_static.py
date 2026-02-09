import os
import subprocess

# Run collectstatic command
result = subprocess.run(['python', 'manage.py', 'collectstatic', '--noinput'], 
                       capture_output=True, text=True)

# Print the output
print(result.stdout)
if result.stderr:
    print("Error:", result.stderr)

# Check if the static directory exists
if os.path.exists('staticfiles'):
    print("Static files collected successfully!")
else:
    print("Failed to collect static files!") 