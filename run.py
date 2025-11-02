#!/usr/bin/env python3
"""
Cross-platform launcher for LLM Benchmark Tool
Works on Windows, Linux, and macOS with a single command: python run.py
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def disable():
        """Disable colors on Windows if not supported"""
        if platform.system() == 'Windows' and not os.environ.get('ANSICON'):
            Colors.HEADER = ''
            Colors.OKBLUE = ''
            Colors.OKCYAN = ''
            Colors.OKGREEN = ''
            Colors.WARNING = ''
            Colors.FAIL = ''
            Colors.ENDC = ''
            Colors.BOLD = ''

def print_header(text):
    """Print a header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def check_python():
    """Check Python version"""
    print_info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} found")
    return True

def check_ollama():
    """Check if Ollama is installed and running"""
    print_info("Checking Ollama installation...")

    # Check if ollama command exists
    if not shutil.which('ollama'):
        print_error("Ollama not found!")
        print_info("Please install Ollama from https://ollama.com/")
        print_info("\nInstallation instructions:")

        os_type = platform.system()
        if os_type == "Linux":
            print_info("  Linux: curl -fsSL https://ollama.com/install.sh | sh")
        elif os_type == "Darwin":
            print_info("  macOS: Download from https://ollama.com/download")
        elif os_type == "Windows":
            print_info("  Windows: Download from https://ollama.com/download")

        return False

    print_success("Ollama is installed")

    # Check if Ollama is running
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success("Ollama service is running")

            # Show available models
            models = [line for line in result.stdout.split('\n') if line.strip() and not line.startswith('NAME')]
            if models:
                print_info(f"Found {len(models)} model(s) installed")
            else:
                print_warning("No models installed yet")
                print_info("Download models with: ollama pull <model-name>")
                print_info("Example: ollama pull deepseek-r1:8b")
            return True
        else:
            print_warning("Ollama is installed but not running")
            print_info("Start Ollama with:")
            if platform.system() == "Windows":
                print_info("  Start the Ollama application")
            else:
                print_info("  ollama serve")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print_warning("Could not check Ollama status")
        print_info("Make sure Ollama is running before benchmarking")
        return False

def setup_venv():
    """Setup virtual environment and install dependencies"""
    venv_dir = Path('.venv')

    # Check if venv exists and is valid
    python_in_venv = venv_dir / ('Scripts/python.exe' if platform.system() == 'Windows' else 'bin/python')
    if venv_dir.exists() and python_in_venv.exists():
        print_info("Virtual environment already exists")
        return True, True  # (success, has_venv)

    print_info("Creating virtual environment...")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'venv', str(venv_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success("Virtual environment created")
            return True, True  # (success, has_venv)
        else:
            # Check if it's the ensurepip issue
            if 'ensurepip' in result.stderr or 'python3-venv' in result.stderr:
                print_warning("Virtual environment module not available")
                print_info("On Debian/Ubuntu: sudo apt install python3-venv")
                print_info("Continuing without virtual environment...")
                # Clean up broken venv
                if venv_dir.exists():
                    import shutil
                    shutil.rmtree(venv_dir, ignore_errors=True)
                return True, False  # (success, no_venv)
            else:
                print_error(f"Failed to create virtual environment: {result.stderr}")
                print_info("Continuing without virtual environment...")
                return True, False  # (success, no_venv)
    except Exception as e:
        print_warning(f"Could not create virtual environment: {e}")
        print_info("Continuing without virtual environment...")
        return True, False  # (success, no_venv)

def get_venv_python(use_venv=True):
    """Get the path to the virtual environment Python"""
    if not use_venv:
        return sys.executable

    venv_dir = Path('.venv')

    if platform.system() == 'Windows':
        python_path = venv_dir / 'Scripts' / 'python.exe'
    else:
        python_path = venv_dir / 'bin' / 'python'

    if python_path.exists():
        return str(python_path)
    return sys.executable

def install_dependencies(use_venv=True):
    """Install Python dependencies"""
    print_info("Installing dependencies...")

    python_path = get_venv_python(use_venv)
    requirements_file = Path('requirements.txt')

    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False

    try:
        # Upgrade pip quietly
        subprocess.run(
            [python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'],
            capture_output=True,
            timeout=60
        )

        # Install requirements
        result = subprocess.run(
            [python_path, '-m', 'pip', 'install', '-r', str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print_success("Dependencies installed")
            return True
        else:
            # Try with --user flag if system install fails
            if 'permission' in result.stderr.lower() or 'externally-managed' in result.stderr.lower():
                print_info("Trying user installation...")
                result = subprocess.run(
                    [python_path, '-m', 'pip', 'install', '--user', '-r', str(requirements_file)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    print_success("Dependencies installed (user)")
                    return True

            print_error(f"Failed to install dependencies")
            print_info("You may need to install manually:")
            print_info(f"  {python_path} -m pip install -r requirements.txt")
            return False
    except subprocess.TimeoutExpired:
        print_error("Installation timed out")
        return False
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False

def check_dependencies(use_venv=True):
    """Check if dependencies are installed"""
    python_path = get_venv_python(use_venv)

    try:
        # Try to import required packages
        result = subprocess.run(
            [python_path, '-c', 'import ollama; import pydantic'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def run_benchmark(args, use_venv=True):
    """Run the benchmark with provided arguments"""
    print_header("Running LLM Benchmark")

    python_path = get_venv_python(use_venv)
    benchmark_script = Path('extended_benchmark.py')

    if not benchmark_script.exists():
        print_error("extended_benchmark.py not found!")
        return False

    # Build command
    cmd = [python_path, str(benchmark_script)] + args

    print_info(f"Command: {' '.join(cmd)}\n")

    # Run the benchmark
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print_warning("\nBenchmark interrupted by user")
        return False
    except Exception as e:
        print_error(f"Failed to run benchmark: {e}")
        return False

def main():
    """Main function"""
    # Disable colors on Windows if needed
    if platform.system() == 'Windows':
        Colors.disable()

    print_header(f"LLM Benchmark Launcher - {platform.system()}")

    # Check prerequisites
    if not check_python():
        sys.exit(1)

    ollama_available = check_ollama()

    # Setup virtual environment
    print_info("\nSetting up environment...")

    success, has_venv = setup_venv()
    if not success:
        print_error("Failed to setup environment")
        sys.exit(1)

    # Install dependencies if needed
    if not check_dependencies(has_venv):
        if not install_dependencies(has_venv):
            print_error("Failed to install dependencies")
            sys.exit(1)
    else:
        print_success("Dependencies already installed")

    # Check if Ollama is available before running
    if not ollama_available:
        print_error("\nCannot run benchmark: Ollama is not available")
        print_info("Please install and start Ollama, then run this script again")
        sys.exit(1)

    # Run the benchmark with any arguments passed to this script
    args = sys.argv[1:]

    if not args:
        print_info("\nNo arguments provided, running with default settings")
        print_info("For quick test, you can use: python run.py --models deepseek-r1:8b --runs-per-prompt 1")
        print_info("For help: python run.py --help\n")

    success = run_benchmark(args, has_venv)

    if success:
        print_success("\nBenchmark completed successfully!")
    else:
        print_error("\nBenchmark failed or was interrupted")
        sys.exit(1)

if __name__ == '__main__':
    main()
