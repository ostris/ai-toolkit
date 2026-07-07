#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# ref https://github.com/runpod/containers/blob/main/container-template/start.sh

# ---------------------------------------------------------------------------- #
#                          Function Definitions                                #
# ---------------------------------------------------------------------------- #

# Replace "/etc/" with "/app/"
# Setup ssh
setup_ssh() {
    if [[ $PUBLIC_KEY ]]; then
        echo "Setting up SSH..."
        mkdir -p ~/.ssh
        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
        chmod 700 -R ~/.ssh

         if [ ! -f /app/ssh/ssh_host_rsa_key ]; then
            ssh-keygen -t rsa -f /app/ssh/ssh_host_rsa_key -q -N ''
            echo "RSA key fingerprint:"
            ssh-keygen -lf /app/ssh/ssh_host_rsa_key.pub
        fi

        if [ ! -f /app/ssh/ssh_host_dsa_key ]; then
            ssh-keygen -t dsa -f /app/ssh/ssh_host_dsa_key -q -N ''
            echo "DSA key fingerprint:"
            ssh-keygen -lf /app/ssh/ssh_host_dsa_key.pub
        fi

        if [ ! -f /app/ssh/ssh_host_ecdsa_key ]; then
            ssh-keygen -t ecdsa -f /app/ssh/ssh_host_ecdsa_key -q -N ''
            echo "ECDSA key fingerprint:"
            ssh-keygen -lf /app/ssh/ssh_host_ecdsa_key.pub
        fi

        if [ ! -f /app/ssh/ssh_host_ed25519_key ]; then
            ssh-keygen -t ed25519 -f /app/ssh/ssh_host_ed25519_key -q -N ''
            echo "ED25519 key fingerprint:"
            ssh-keygen -lf /app/ssh/ssh_host_ed25519_key.pub
        fi

        service ssh start

        echo "SSH host keys:"
        for key in /app/ssh/*.pub; do
            echo "Key: $key"
            ssh-keygen -lf $key
        done
    fi
}

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /app/rp_environment
    echo 'source /app/rp_environment' >> ~/.bashrc
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #


echo "Pod Started"

setup_ssh
export_env_vars
echo "Starting AI Toolkit UI..."
cd /app/ui && npm run start 