#!/usr/bin/env python3
"""
Test script to verify .env file loading for admin panel
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from parent directory (same as admin_panel.py)
env_path = Path(__file__).parent.parent / '.env'
print(f"Loading .env from: {env_path}")
print(f".env exists: {env_path.exists()}")

load_dotenv(dotenv_path=env_path)

# Test loading credentials
ADMIN_USER = os.getenv('ADMIN_USER')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
SECRET_KEY = os.getenv('SECRET_KEY')

print("\n" + "="*60)
print("🔐 Environment Variables Test")
print("="*60)
print(f"ADMIN_USER: {ADMIN_USER}")
print(f"ADMIN_PASSWORD: {ADMIN_PASSWORD}")
print(f"SECRET_KEY: {SECRET_KEY[:20]}..." if SECRET_KEY else "SECRET_KEY: NOT SET")
print("="*60)

# Test credentials
test_username = "agamadmin"
test_password = "agampassword"

print(f"\n✅ Test credentials:")
print(f"   Test username: {test_username}")
print(f"   Test password: {test_password}")
print(f"   Loaded username: {ADMIN_USER}")
print(f"   Loaded password: {ADMIN_PASSWORD}")
print(f"   Username match: {test_username == ADMIN_USER}")
print(f"   Password match: {test_password == ADMIN_PASSWORD}")

if test_username == ADMIN_USER and test_password == ADMIN_PASSWORD:
    print("\n✅ SUCCESS: Credentials match!")
else:
    print("\n❌ FAIL: Credentials do not match!")
    print("\nPossible issues:")
    print("1. .env file not found")
    print("2. Environment variables not set correctly")
    print("3. Whitespace or special characters in .env values")
