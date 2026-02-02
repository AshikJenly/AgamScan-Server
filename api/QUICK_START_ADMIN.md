# Quick Start Guide - Admin Panel with Login

## 🚀 Starting the Services

### 1. Start the Main API (Port 8005)
```bash
cd /home/ashikjenly/Work/AgamScan-Server/api
python app.py
```

### 2. Start the Admin Panel (Port 8006)
```bash
cd /home/ashikjenly/Work/AgamScan-Server/api
python admin_panel.py
```

---

## 🔐 Admin Panel Access

### Login Credentials (from .env)
- **URL**: http://localhost:8006/login
- **Username**: `agamadmin`
- **Password**: `agampassword`

### Steps:
1. Open browser to http://localhost:8006
2. You'll be redirected to login page
3. Enter credentials
4. Click Login
5. Access the admin panel dashboard

---

## 🧪 Testing the is_final Parameter

### Test 1: Preview Mode (Hybrid Detection)
```bash
curl -X POST "http://localhost:8005/process" \
  -F "file=@test_card.jpg" \
  -F "is_final=false"
```

**Expected Behavior:**
- Uses hybrid finger detection (area difference + MediaPipe)
- Logs show: `"is_final": false`
- Admin panel shows: 👁️ **PREVIEW** badge
- Finger check uses both methods

### Test 2: Final Mode (MediaPipe Only)
```bash
curl -X POST "http://localhost:8005/process" \
  -F "file=@test_card.jpg" \
  -F "is_final=true"
```

**Expected Behavior:**
- Uses MediaPipe-only finger detection
- Logs show: `"is_final": true`
- Admin panel shows: 🎯 **FINAL** badge
- Finger check skips area difference method

---

## 📊 Viewing Results in Admin Panel

### 1. Dashboard Statistics
- Total Requests
- Passed/Failed counts
- Success rate

### 2. Request List
- See all requests with badges
- Filter by status (all/passed/failed)
- Sort by timestamp (newest first)

### 3. Request Details
Click any request to see:
- **Basic Info**: filename, timestamp, stage, **request type (is_final)**
- **Quality Checks**: blur, glare, finger (with detection mode)
- **Finger Details**:
  - Detection mode (MediaPipe Only / Hybrid)
  - Fingertips detected
  - Fingertip count
  - Area difference ratio
- **OCR Results**: extracted text
- **NER Results**: structured fields
- **Images**: original and annotated

---

## 🔍 Understanding Finger Detection in Admin Panel

### Preview Mode (is_final=false):
```
Detection Mode: 🔄 Hybrid Detection
- Area difference check: ✅ Performed
- MediaPipe check: ✅ Performed
- Fingertips: Shows if detected
```

### Final Mode (is_final=true):
```
Detection Mode: 🖐️ MediaPipe Only
- Area difference check: ⏭️ Skipped
- MediaPipe check: ✅ Performed
- Fingertips: Shows if detected
```

---

## 🐛 Troubleshooting

### Cannot access admin panel
- **Issue**: Getting redirected to login
- **Solution**: Make sure you're logged in at /login

### Login not working
- **Issue**: Invalid credentials
- **Check**: Verify .env file has correct USER and PASSWORD
- **Solution**: Use `agamadmin` / `agampassword`

### is_final not showing in logs
- **Issue**: Old requests before update
- **Solution**: Send new requests after implementation

### Detection mode not showing
- **Issue**: Missing finger metrics
- **Check**: Ensure MediaPipe is initialized
- **Solution**: Check console for MediaPipe initialization messages

---

## 📝 Environment Variables

### Required in .env:
```properties
# Admin Panel Authentication
USER=agamadmin
PASSWORD=agampassword
SECRET_KEY=agamscan-secret-key-change-in-production-2026

# API Configuration
API_HOST=0.0.0.0
API_PORT=8005
```

---

## 🔐 Changing Admin Credentials

### 1. Edit .env file:
```bash
nano /home/ashikjenly/Work/AgamScan-Server/.env
```

### 2. Update credentials:
```properties
USER=your_new_username
PASSWORD=your_new_password
SECRET_KEY=your-random-secret-key-here
```

### 3. Restart admin panel:
```bash
# Stop current process (Ctrl+C)
python admin_panel.py
```

---

## 🚀 Production Deployment

### Security Checklist:
- [ ] Change SECRET_KEY to random strong value
- [ ] Change default admin password
- [ ] Enable HTTPS
- [ ] Add rate limiting
- [ ] Add password hashing (bcrypt)
- [ ] Add session timeout
- [ ] Add CSRF protection
- [ ] Restrict admin panel to internal network
- [ ] Add audit logging
- [ ] Set proper file permissions on .env

### Recommended SECRET_KEY Generation:
```python
import secrets
print(secrets.token_urlsafe(32))
```

---

## 📖 API Endpoints

### Main API (Port 8005)
- `POST /process` - Process document
  - Parameters: `file`, `is_final` (bool)
- `GET /health` - Health check
- `GET /config` - Get configuration

### Admin Panel (Port 8006)
- `GET /login` - Login page
- `POST /login` - Authenticate
- `GET /logout` - Logout
- `GET /` - Admin dashboard (protected)
- `GET /api/requests` - Get all requests (protected)
- `GET /api/request/<id>` - Get request details (protected)
- `GET /api/statistics` - Get statistics (protected)

---

## 💡 Tips

### Logout
Click the 🚪 **Logout** button in the top-right corner

### Auto Refresh
Click ⏱️ **Auto Refresh (30s)** to enable automatic dashboard updates

### Filtering
Use the filter dropdown to show only passed or failed requests

### Viewing Images
Click any request to see original and annotated images

### Export Data
Right-click on detail modal and "Save As" to export JSON data

---

## 📞 Support

If you encounter issues:
1. Check console logs for errors
2. Verify all services are running
3. Check .env file configuration
4. Ensure all dependencies are installed
5. Review ADMIN_PANEL_LOGIN_IMPLEMENTATION.md for detailed info

---

**Last Updated**: February 2, 2026  
**Version**: 2.0
