# ⚡ Quick Start Guide

Get the AI Chatbot running in 3 minutes!

---

## 🖥️ Desktop Quick Start

### Windows

1. **Download**
   ```
   Download from GitHub: Clone or Download ZIP
   Extract to a folder
   ```

2. **Install**
   ```cmd
   Double-click: install.bat
   Wait for installation to complete
   ```

3. **Run**
   ```cmd
   python launch_chatbot.py server
   ```

4. **Open Browser**
   ```
   http://localhost:5000
   ```

**Done!** Start chatting! 🎉

---

### Linux / macOS

1. **Download**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI_timestamp_context.git
   cd AI_timestamp_context
   ```

2. **Install**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Run**
   ```bash
   python launch_chatbot.py server
   ```

4. **Open Browser**
   ```
   http://localhost:5000
   ```

**Done!** Start chatting! 🎉

---

## 📱 Mobile Quick Start (Android)

### Option A: Access Remote Server (Easiest)

1. **Start server on desktop**
   ```bash
   python launch_chatbot.py server --host 0.0.0.0
   ```

2. **Find desktop IP**
   ```bash
   # Windows
   ipconfig

   # Linux/Mac
   ifconfig | grep inet
   ```

3. **Open on phone**
   ```
   http://YOUR_DESKTOP_IP:5000
   ```

**Done!** Chat from your phone! 📱

### Option B: Run on Android (Termux)

1. **Install Termux**
   - Download from F-Droid or GitHub

2. **Setup**
   ```bash
   pkg install python git
   git clone <repo>
   cd AI_timestamp_context
   pip install -r requirements.txt
   ```

3. **Run**
   ```bash
   python launch_chatbot.py cli
   ```

**Done!** Chat in Termux! 🤖

---

## 🎯 What to Try First

### Web Interface

1. Type: "Hello! What can you do?"
2. Get a response
3. Click 👍 or 👎 to give feedback
4. Watch the bot learn!

### CLI Mode

```bash
python launch_chatbot.py cli --feedback
```

Type your messages, get responses, provide feedback.

---

## 🚀 Next Steps

### Enable Learning

```bash
# In a separate terminal
python launch_chatbot.py train --quick-minutes 5
```

The bot now learns every 5 minutes!

### Try Different Modes

```bash
# Web interface
python launch_chatbot.py server

# Command line
python launch_chatbot.py cli

# REST API
python launch_chatbot.py api

# Training only
python launch_chatbot.py train --manual-steps 10
```

### Customize Settings

Edit `config.py` or create `my_config.json`:

```json
{
  "chatbot": {
    "model_name": "microsoft/DialoGPT-medium",
    "temperature": 0.8
  },
  "rag": {
    "retrieval_top_k": 10
  }
}
```

Then:
```bash
python launch_chatbot.py server --config my_config.json
```

---

## 📊 Check Status

### Web Dashboard

Open http://localhost:5000 and see:
- Conversations count
- Knowledge items
- Confidence scores

### CLI Stats

```bash
python launch_chatbot.py cli
```

Then type: `stats`

### API Stats

```bash
curl http://localhost:5000/api/stats
```

---

## 🎓 Common Use Cases

### Customer Support Bot

```bash
python launch_chatbot.py server
# Load support documentation via web interface
```

### Learning Assistant

```bash
python launch_chatbot.py cli --feedback
# Ask questions, give feedback, watch it improve
```

### Research Bot

```bash
# Use high retrieval settings
python launch_chatbot.py server
# Feed it research papers via chat
```

---

## 🔧 Troubleshooting

### Problem: Installation fails

**Solution:**
```bash
# Use Python 3.8+
python --version

# Upgrade pip
pip install --upgrade pip

# Install manually
pip install torch transformers flask
```

### Problem: Out of memory

**Solution:**
Use smaller model in `config.py`:
```python
model_name = "microsoft/DialoGPT-small"  # Use small not medium
```

### Problem: Slow responses

**Solution:**
- Use CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Or run on cloud server and access via browser

### Problem: Can't access from phone

**Solution:**
```bash
# Use 0.0.0.0 not localhost
python launch_chatbot.py server --host 0.0.0.0

# Check firewall allows port 5000
```

---

## 📚 Learn More

- **Full Documentation**: `CHATBOT.md`
- **All Features**: `FEATURES.md`
- **Mobile Guide**: `MOBILE.md`
- **API Reference**: `api_server.py`

---

## 🎉 You're Ready!

The chatbot is now:
- ✅ Running locally
- ✅ Learning from interactions
- ✅ Storing knowledge
- ✅ Accessible from browser

**Start chatting and watch it learn!** 🤖💬

---

## 💡 Pro Tips

1. **Give Feedback**: 👍👎 buttons help it learn faster
2. **Use Training Mode**: Background learning improves responses
3. **Save Checkpoints**: Auto-saved every 6 hours
4. **Access Anywhere**: Use `--host 0.0.0.0` for network access
5. **Mobile Friendly**: Web UI works great on phones

---

**Need help?** Check `CHATBOT.md` or open an issue on GitHub!

Happy chatting! 🚀
