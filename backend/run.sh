#!/usr/bin/env bash
# Start script for Render deployment

set -o errexit  # Exit on error

# Start Gunicorn
gunicorn backend.wsgi:application --log-file - 