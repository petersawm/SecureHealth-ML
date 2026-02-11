"""
Automated setup script for development environment.

"""
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


class SetupManager:
    """Manage project setup and installation."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / "venv"
        self.requirements_path = self.project_root / "requirements.txt"
    
    def check_python_version(self) -> Tuple[bool, str]:
        """
        Check if Python version is 3.9+.
        
        Returns:
            (is_valid, version_string)
        """
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        is_valid = version.major >= 3 and version.minor >= 9
        return is_valid, version_str
    
    def create_virtual_environment(self) -> bool:
        """
        Create virtual environment.
        
        Returns:
            True if successful
        """
        try:
            print("\n Creating virtual environment...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True,
                capture_output=True
            )
            print(" Virtual environment created")
            return True
        except subprocess.CalledProcessError as e:
            print(f" Failed to create venv: {e}")
            return False
    
    def get_pip_path(self) -> Path:
        """Get path to pip executable in venv."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"
    
    def install_dependencies(self) -> bool:
        """
        Install required packages.
        
        Returns:
            True if successful
        """
        try:
            print("\n Installing dependencies...")
            pip_path = self.get_pip_path()
            
            # Upgrade pip first
            subprocess.run(
                [str(pip_path), "install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )
            
            # Install requirements
            subprocess.run(
                [str(pip_path), "install", "-r", str(self.requirements_path)],
                check=True
            )
            print(" Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f" Failed to install dependencies: {e}")
            return False
    
    def verify_installation(self) -> Tuple[bool, List[str]]:
        """
        Verify critical packages are installed.
        
        Returns:
            (all_ok, missing_packages)
        """
        critical_packages = ["torch", "flwr", "opacus", "numpy", "sklearn"]
        missing = []
        
        print("\n Verifying installation...")
        for package in critical_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package}")
                missing.append(package)
        
        return len(missing) == 0, missing
    
    def run(self) -> bool:
        """
        Run complete setup process.
        
        Returns:
            True if setup successful
        """
        print("=" * 60)
        print("SecureHealth-ML Development Setup")
        print("=" * 60)
        
        # Check Python version
        is_valid, version = self.check_python_version()
        print(f"\n Python version: {version}")
        if not is_valid:
            print(" Python 3.9+ required")
            return False
        print(" Python version OK")
        
        # Create venv
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Verify installation
        all_ok, missing = self.verify_installation()
        if not all_ok:
            print(f"\n Missing packages: {', '.join(missing)}")
            return False
        
        print("\n" + "=" * 60)
        print(" Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Activate environment:")
        if sys.platform == "win32":
            print("     venv\\Scripts\\activate")
        else:
            print("     source venv/bin/activate")
        print("  2. Run tests:")
        print("     pytest tests/")
        print("  3. Start development!")
        print("=" * 60)
        
        return True


def main():
    """Main entry point."""
    manager = SetupManager()
    success = manager.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()