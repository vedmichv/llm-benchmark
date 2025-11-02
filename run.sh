#!/bin/bash
# Cross-platform launcher for LLM Benchmark (Linux/macOS)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if Python 3 is installed
check_python() {
    print_info "Checking Python installation..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        print_info "Please install Python 3.8 or higher"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
}

# Check if Ollama is installed
check_ollama() {
    print_info "Checking Ollama installation..."

    if ! command -v ollama &> /dev/null; then
        print_error "Ollama is not installed"
        print_info "Install from: https://ollama.com/"
        print_info "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
        print_info "  macOS: Download from https://ollama.com/download"
        return 1
    fi

    print_success "Ollama is installed"

    # Check if Ollama is running
    if ollama list &> /dev/null; then
        print_success "Ollama service is running"
        return 0
    else
        print_warning "Ollama is installed but not running"
        print_info "Start with: ollama serve"
        return 1
    fi
}

# Main execution
main() {
    print_header "LLM Benchmark Launcher - $(uname -s)"

    # Check prerequisites
    check_python

    OLLAMA_AVAILABLE=0
    if check_ollama; then
        OLLAMA_AVAILABLE=1
    fi

    # Change to script directory
    cd "$(dirname "$0")"

    # Run the Python launcher which handles the rest
    print_info "Starting benchmark launcher..."
    python3 run.py "$@"
}

main "$@"
