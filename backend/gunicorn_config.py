import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes - use fewer workers to prevent memory issues
workers = 2  # Reduced from multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 300  # Increased timeout to 5 minutes
keepalive = 2

# Memory settings
max_requests = 1000
max_requests_jitter = 50
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'deepscalers'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None 