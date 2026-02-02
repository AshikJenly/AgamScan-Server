#!/bin/bash
# Test admin panel login

echo "🧪 Testing Admin Panel Login"
echo "================================"
echo ""

# Test login
echo "📝 Sending login request..."
response=$(curl -s 'http://localhost:8006/login' \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-raw '{"username":"agamadmin","password":"agampassword"}')

echo "Response: $response"
echo ""

# Check if login was successful
if echo "$response" | grep -q '"success": true'; then
    echo "✅ Login successful!"
else
    echo "❌ Login failed!"
    echo ""
    echo "Possible issues:"
    echo "1. Admin panel is not running (python admin_panel.py)"
    echo "2. Credentials don't match .env file"
    echo "3. .env file not loaded correctly"
fi

echo ""
echo "================================"
