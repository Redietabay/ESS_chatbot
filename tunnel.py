"""
Run this in a SECOND terminal window while app.py is running in the first.
It opens a public URL that tunnels to your local chatbot on port 5000.

Usage:
    (venv) PS C:\\Users\\toshiba\\ESS_chatbot> python tunnel.py
"""
from pyngrok import ngrok

public_url = ngrok.connect(5000)
print("=" * 60)
print(f"Your PUBLIC presentation URL is:")
print(f"  {public_url}")
print("=" * 60)
print("Keep this terminal window open during your presentation.")
print("Press CTRL+C here to stop the tunnel when you're done.")

input("\nPress Enter to close the tunnel...\n")
ngrok.disconnect(public_url)
