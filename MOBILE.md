# 📱 Mobile Installation Guide

## Running AI Chatbot on Android

You can run the full AI chatbot system on Android devices using Termux!

---

## 🚀 Quick Install (Android)

### Method 1: Termux (Recommended)

**Step 1: Install Termux**
- Download from F-Droid: https://f-droid.org/en/packages/com.termux/
- Or GitHub: https://github.com/termux/termux-app/releases

**Step 2: Setup Termux**
```bash
# Update packages
pkg update && pkg upgrade

# Install required packages
pkg install python git wget

# Install pip
pip install --upgrade pip
```

**Step 3: Clone and Install**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/AI_timestamp_context.git
cd AI_timestamp_context

# Install dependencies (lightweight for mobile)
pip install numpy networkx matplotlib-base pandas python-dateutil pyyaml psutil schedule

# Install ML (choose one based on storage)
# Option A: CPU-only PyTorch (smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Option B: Skip PyTorch, use basic mode
# (The system will work without neural features)

# Install web framework
pip install flask flask-cors flask-socketio python-socketio eventlet

# Install transformers
pip install transformers sentence-transformers
```

**Step 4: Run the Chatbot**
```bash
# Command-line mode (lightweight)
python launch_chatbot.py cli

# Web interface (access from phone browser)
python launch_chatbot.py server --host 0.0.0.0 --port 8080

# Then open: http://localhost:8080 in phone browser
```

---

### Method 2: Access via Browser (Any Device)

If you have the chatbot running on a desktop/server:

**Step 1: Start server with network access**
```bash
# On your desktop/server
python launch_chatbot.py server --host 0.0.0.0 --port 5000
```

**Step 2: Find your server IP**
```bash
# Linux/Mac
ifconfig | grep "inet "

# Windows
ipconfig
```

**Step 3: Access from mobile**
- Open browser on Android
- Navigate to: `http://YOUR_SERVER_IP:5000`
- Example: `http://192.168.1.100:5000`

---

## 🔧 Termux Optimization Tips

### Reduce Memory Usage

Create `config_mobile.py`:
```python
from neural_chatbot import ChatbotConfig

# Lightweight config for mobile
mobile_config = ChatbotConfig(
    model_name="microsoft/DialoGPT-small",  # Use smallest model
    batch_size=2,  # Smaller batches
    experience_replay_size=100,  # Less memory
    episodic_memory_size=50,
    semantic_memory_size=100,
    max_length=256  # Shorter sequences
)
```

Then modify `launch_chatbot.py` to use mobile config when on Android.

### Use CLI Mode (Lightest)

```bash
python launch_chatbot.py cli
```
This uses minimal resources and works great on mobile!

### Storage Tips

```bash
# Check storage
df -h

# Clean pip cache
pip cache purge

# Remove unused packages
pkg autoremove
```

---

## 📱 Building Android APK (Advanced)

### Using Buildozer (Python to APK)

**Step 1: Install Buildozer**
```bash
pip install buildozer
```

**Step 2: Create buildozer.spec**
See `buildozer.spec` file in the repository.

**Step 3: Build APK**
```bash
buildozer android debug
```

**Result**: `bin/aichatbot-0.1-debug.apk`

---

## 🌐 Web Access from Any Device

The easiest way to use on mobile:

1. Run server on your computer:
   ```bash
   python launch_chatbot.py server
   ```

2. Get your computer's IP address

3. On your phone, open browser and go to:
   ```
   http://YOUR_COMPUTER_IP:5000
   ```

4. Bookmark it for easy access!

---

## 💡 Mobile-Optimized Features

The web interface is already mobile-responsive:
- ✅ Touch-friendly buttons
- ✅ Responsive layout
- ✅ Swipe gestures
- ✅ Mobile keyboard support
- ✅ Fullscreen mode

---

## 📊 Performance on Mobile

### Minimum Requirements:
- **RAM**: 2GB (4GB recommended)
- **Storage**: 1GB free space
- **Android**: 7.0+ (for Termux)

### Performance:
- **DialoGPT-small**: ~10-30 seconds per response (on Termux)
- **CLI Mode**: Instant (no model loading)
- **Web Access**: 1-3 seconds (when server on desktop)

### Recommended Setup:
- Run server on desktop/cloud
- Access via mobile browser
- Best performance with lowest battery drain!

---

## 🔋 Battery Optimization

```bash
# Run in background
nohup python launch_chatbot.py server > chatbot.log 2>&1 &

# Stop background process
pkill -f launch_chatbot
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory
**Solution**: Use smaller model or CLI mode
```bash
# Edit config to use tiny model
model_name="microsoft/DialoGPT-small"  # or skip neural features
```

### Issue: Slow Response
**Solution**: Use remote server instead
```bash
# On desktop
python launch_chatbot.py server --host 0.0.0.0

# Access from mobile browser
```

### Issue: Package Install Fails
**Solution**: Try minimal installation
```bash
# Core only
pip install flask numpy pandas

# Skip ML libraries
# Use basic conversation mode without neural features
```

### Issue: Termux Permission Denied
**Solution**: Grant storage permission
```bash
termux-setup-storage
```

---

## 📦 Lightweight Alternative

For very limited devices, use **CLI mode without neural features**:

```python
# simplified_chat.py
from conversation_system import EnhancedConversationSystem
from datetime import datetime

system = EnhancedConversationSystem()

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    result = system.process_message(
        user_input,
        datetime.utcnow(),
        "user"
    )

    print(f"Bot: Processing complete at depth {result['context_depth']}")
```

Run:
```bash
python simplified_chat.py
```

---

## 🌟 Best Practices for Mobile

1. **Use Web Browser Access** (easiest, fastest)
2. **Run server elsewhere** (desktop/cloud)
3. **Use CLI mode** for Termux (lightest)
4. **Enable wake lock** in Termux for background
5. **Close other apps** when using Termux
6. **Use WiFi** not mobile data (saves battery)

---

## 📱 PWA (Progressive Web App)

The web interface can be added to home screen:

**iOS/Android**:
1. Open `http://YOUR_SERVER:5000` in browser
2. Tap "Share" / "Menu"
3. Select "Add to Home Screen"
4. Icon will appear on home screen!

---

## 🚀 Cloud Hosting (Recommended for Mobile)

Deploy once, access everywhere:

**Free Options:**
- Heroku (free tier)
- Render (free tier)
- Railway (free tier)
- PythonAnywhere (free tier)

Then access from mobile browser with zero setup!

See deployment guide in `CHATBOT.md`.

---

## 📞 Support

For mobile-specific issues:
1. Check Termux logs
2. Try smaller model
3. Use web access instead
4. Report issues on GitHub

---

**Mobile access made easy!** 📱✨

