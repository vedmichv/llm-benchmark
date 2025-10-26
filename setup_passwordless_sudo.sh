#!/bin/bash
# Setup passwordless sudo for Ollama model offloading
# This allows the benchmark script to offload models without password prompts

echo "================================================"
echo "Ollama Benchmark - Passwordless Sudo Setup"
echo "================================================"
echo ""
echo "This script will configure sudo to allow passwordless execution"
echo "of commands needed for model offloading:"
echo "  - pkill (to stop ollama serve)"
echo "  - journalctl (to read ollama logs)"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Get current user
CURRENT_USER=$(whoami)

# Create sudoers configuration
SUDOERS_FILE="/etc/sudoers.d/ollama-benchmark"

echo ""
echo "Creating sudoers configuration..."

# Create the configuration
sudo tee "$SUDOERS_FILE" > /dev/null << EOF
# Ollama Benchmark - Passwordless sudo configuration
# Created: $(date)
# User: $CURRENT_USER

# Allow user to kill ollama processes for model offloading
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/pkill -f ollama serve
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/pkill ollama

# Allow user to read ollama logs for diagnostics
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/journalctl -u ollama*
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/tail -n * /proc/*/fd/*

# Allow user to check process status
$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/pgrep *
EOF

# Set correct permissions
sudo chmod 0440 "$SUDOERS_FILE"

# Verify syntax
if sudo visudo -c -f "$SUDOERS_FILE" > /dev/null 2>&1; then
    echo "✓ Sudoers configuration created successfully"
    echo ""
    echo "File location: $SUDOERS_FILE"
    echo ""
    echo "You can now run the benchmark without sudo password prompts:"
    echo "  python3 extended_benchmark.py"
    echo ""
    echo "To remove this configuration later:"
    echo "  sudo rm $SUDOERS_FILE"
    echo ""
else
    echo "✗ Error: Invalid sudoers syntax"
    sudo rm "$SUDOERS_FILE"
    exit 1
fi

echo "================================================"
echo "Setup complete!"
echo "================================================"
