# VillageForge AI — Complete Setup Guide
# Kaggle × Google DeepMind Gemma 4 Good Hackathon

===========================================================
 WHAT YOU NEED (Summary)
===========================================================

| Service          | Cost  | API Key? | Time to set up |
|------------------|-------|----------|----------------|
| Google Earth Engine | Free (non-commercial) | Yes (OAuth) | ~10 min |
| OpenCage Geocoder   | Free (2,500/day)      | Yes (string key) | 2 min  |
| Ollama + Gemma 4    | Free (local)          | No              | 15 min |
| OpenStreetMap Overpass | Free              | No              | 0 min  |
| ISRIC SoilGrids     | Free                  | No              | 0 min  |
| NASA POWER API      | Free                  | No              | 0 min  |

Total cost: $0. Total setup time: ~30 minutes.

===========================================================
 STEP 1: GOOGLE EARTH ENGINE
===========================================================

Earth Engine gives you free access to petabytes of satellite data.

1a. Create a Google Cloud Project
    → Go to: https://console.cloud.google.com/
    → Click "New Project"
    → Name it e.g. "villageforge-ai"
    → Note your PROJECT ID (e.g. "villageforge-ai-123456")

1b. Enable the Earth Engine API
    → Go to: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
    → Click "Enable"

1c. Register for Earth Engine Non-Commercial use
    → Go to: https://code.earthengine.google.com/register
    → Select "Unpaid usage" → "Academic / Research"
    → Link to your Google Cloud project from step 1a

1d. Authenticate on your machine
    Run in terminal:
        earthengine authenticate
    This opens a browser → sign in → copy the token back.

1e. Set your project in .env file:
    EE_PROJECT=your-project-id-here

===========================================================
 STEP 2: OPENCAGE GEOCODER API KEY
===========================================================

Converts pincode/ZIP → latitude + longitude.
Free tier: 2,500 requests/day (plenty for demo).

2a. Sign up at: https://opencagedata.com/api
2b. After email verification, go to Dashboard
2c. Copy your API key (looks like: 97093d6d325442379ab...)
2d. Set in .env file:
    OPENCAGE_API_KEY=your_key_here

===========================================================
 STEP 3: GEMMA 4 VIA OLLAMA (LOCAL, OFFLINE)
===========================================================

This replaces GPT-4. Gemma 4 runs on YOUR machine.
No API key needed. No data leaves your computer.

3a. Install Ollama
    → Mac/Linux:
        curl -fsSL https://ollama.com/install.sh | sh
    → Windows:
        Download from: https://ollama.com/download/windows

3b. Pull Gemma 4 model
    Choose based on your hardware:

    For 16GB+ RAM / good GPU:
        ollama pull gemma3:27b

    For 8–16GB RAM:
        ollama pull gemma3:12b

    For 4–8GB RAM (lightweight demo):
        ollama pull gemma3:4b

    This downloads the model (~15–55 GB). Do this on WiFi.
    After download, it works OFFLINE.

3c. Verify Ollama is running
        ollama serve
    (It starts automatically on Mac/Windows)

3d. Test it works:
        curl http://localhost:11434/api/generate -d '{
          "model": "gemma3:27b",
          "prompt": "Say hello",
          "stream": false
        }'

3e. Set in village_engine.py:
    GEMMA_MODEL = "gemma3:27b"   # match whichever you pulled

3f. If running Ollama on a different machine:
    Set in .env:
    OLLAMA_BASE_URL=http://192.168.1.x:11434

===========================================================
 STEP 4: NO-KEY SERVICES (automatic)
===========================================================

These work out of the box — no sign-up needed:

• OpenStreetMap Overpass API
    Used for: nearest roads, hospitals, clinics
    URL: https://overpass-api.de/api/interpreter
    Limit: ~10,000 requests/day, rate-limited gracefully

• ISRIC SoilGrids
    Used for: soil pH, organic carbon, clay, nitrogen
    URL: https://rest.isric.org/soilgrids/v2.0/
    Limit: None for basic queries

• NASA POWER API
    Used for: solar irradiance validation
    URL: https://power.larc.nasa.gov/api/
    Limit: None for point queries

===========================================================
 STEP 5: SET UP ENVIRONMENT FILE
===========================================================

Create a file named .env in the villageforge/ folder:

    # .env
    OPENCAGE_API_KEY=your_opencage_key_here
    EE_PROJECT=your-gcloud-project-id
    OLLAMA_BASE_URL=http://localhost:11434

Then in Python, load it:
    from dotenv import load_dotenv
    load_dotenv()

===========================================================
 STEP 6: INSTALL PYTHON DEPENDENCIES
===========================================================

Requires Python 3.10 or higher.

    # Create virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate        # Mac/Linux
    venv\Scripts\activate           # Windows

    # Install packages
    pip install -r requirements.txt

===========================================================
 STEP 7: RUN THE APP
===========================================================

Option A — Web UI (recommended for demo):
    python app.py
    → Opens at http://localhost:7860

Option B — Command line:
    python village_engine.py
    → Prompts for pincode, generates PDF

Option C — Kaggle Notebook:
    Upload village_engine.py as a dataset
    Use Kaggle's internet-enabled environment
    Note: Ollama won't run on Kaggle; use Kaggle's
          hosted Gemma API instead (see below)

===========================================================
 KAGGLE NOTEBOOK ALTERNATIVE (no local Ollama)
===========================================================

If running directly on Kaggle, replace the call_gemma()
function with Kaggle's hosted inference:

    import kaggle_secrets
    # Kaggle provides Gemma 4 inference via their platform
    # See: kaggle.com/competitions/gemma-4-good-hackathon

    # Or use Google AI Studio (free, Gemma 4 available):
    # https://aistudio.google.com/
    # Model: gemma-3-27b-it
    # API Key: from aistudio.google.com → Get API Key
    # Set: GOOGLE_AI_KEY=your_key_here

===========================================================
 HARDWARE REQUIREMENTS
===========================================================

Minimum (gemma3:4b):
    → 8 GB RAM, any modern CPU
    → ~4 GB disk for model
    → Analysis time: ~2–3 minutes

Recommended (gemma3:12b):
    → 16 GB RAM, or 8 GB VRAM GPU
    → ~8 GB disk for model
    → Analysis time: ~45–90 seconds

Best (gemma3:27b):
    → 32 GB RAM, or 16 GB VRAM GPU
    → ~18 GB disk for model
    → Analysis time: ~30–60 seconds

===========================================================
 TROUBLESHOOTING
===========================================================

"Earth Engine not initialized"
    → Run: earthengine authenticate
    → Make sure EE_PROJECT is set in .env

"Could not geocode the pincode"
    → Check OPENCAGE_API_KEY in .env
    → Try the full address instead: "Sitamarhi, Bihar, India"

"Connection refused to Ollama"
    → Run: ollama serve (in a separate terminal)
    → Check OLLAMA_BASE_URL in .env

"SoilGrids returned no data"
    → Normal for ocean/ice locations — soil layer gracefully skips

"Overpass API timeout"
    → Normal during high load — connectivity layer returns None values
    → Plan still generates using other layers

===========================================================
 KAGGLE SUBMISSION CHECKLIST
===========================================================

✅ Working demo (Gradio UI at localhost:7860)
✅ Public code repository (push to GitHub before submitting)
✅ Technical write-up (explain Gemma 4 usage + data layers)
✅ Short video (~2 min: enter pincode → show plan → show PDF)
✅ Uses Gemma 4 as core reasoning model
✅ Works offline after initial setup
✅ Addresses education + health + climate focus areas
✅ Submission deadline: May 18, 2026, 23:59 UTC

===========================================================
 FILE STRUCTURE
===========================================================

villageforge/
├── village_engine.py    ← Main backend (all data layers + Gemma 4)
├── app.py               ← Gradio web UI
├── requirements.txt     ← Python dependencies
├── SETUP.md             ← This file
├── .env                 ← Your keys (DO NOT commit to GitHub)
├── .gitignore           ← Add .env here!
└── outputs/             ← Generated PDFs and JSONs
