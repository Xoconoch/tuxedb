#!/bin/bash
set -e

# Set default UID and GID if not provided
USER_ID=${UID:-1000}
GROUP_ID=${GID:-1000}

# Create the group if it doesn't exist
if ! getent group appgroup > /dev/null; then
    groupadd -g "$GROUP_ID" appgroup
fi

# Create the user if it doesn't exist
if ! id -u appuser > /dev/null 2>&1; then
    useradd -m -u "$USER_ID" -g appgroup -s /bin/bash appuser
fi

# Change ownership of the app directory
chown -R appuser:appgroup /app

# Execute the command as the specified user
exec gosu appuser "$@"
