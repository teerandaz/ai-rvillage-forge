# 🌍 VillageForge AI (v3.0)
**Multimodal, Agentic, Open-Data Infrastructure Planner for Social Good**

VillageForge AI is an intelligent planning assistant designed to help NGOs, governments, and rural developers plan critical infrastructure (like schools, hospitals, and solar microgrids) in remote or underserved areas. 

Instead of relying on guesswork, VillageForge aggregates open-source satellite data, global health databases, and climate models, then uses a panel of AI agents to debate and design a highly resilient, data-driven project plan.

---

## 🧠 How It Works (In Simple Terms)

1. **You make a request:** You can type your request or speak into your microphone in **any language**. Tell the AI what you want to build, where (e.g., a pincode or village name), and your budget.
2. **The AI gathers data:** The engine reaches out to free, public databases (NASA, Google Earth Engine, OpenStreetMap, World Bank, etc.) to check the location's flood risk, solar potential, population density, road access, and more.
3. **The AI Agents debate:** Behind the scenes, different AI "personas" look at the data. A strict *Structural Engineer* checks for earthquake and flood risks. A *Social Advocate* checks if the clinic is too far for the elderly. A *Climate Scientist* checks 2050 heatwave projections. They argue and fix the plan until it's safe.
4. **You get a complete report:** The app generates a 3D risk map, a 5-year satellite timelapse, a budget breakdown, an interactive timeline, and a downloadable PDF report. 
5. **Chat & Listen:** You can chat directly with your generated plan to ask specific questions, or click a button to hear the summary read out loud in your local language!

---

## ✨ Key Features
- **🎙️ Multilingual Voice Input:** Speak in Hindi, Spanish, Swahili, etc. The AI translates it to English instantly using OpenAI's Whisper.
- **🛰️ Satellite & Open Data:** Pulls real-time data for topography, weather, soil quality, and air pollution without requiring paid API keys.
- **🤖 Multi-Agent Debate:** Plans are stress-tested by AI personas to ensure safety and budget efficiency.
- **🗺️ Interactive 3D & 2D Mapping:** Visualize flood risks, population clusters, and local terrain in your browser.
- **💬 Plan Chatbot & Audio:** Talk to your infrastructure plan or have it read out loud in multiple languages using `gTTS`.
- **📄 Auto-Generated PDFs:** Instantly export professional, boardroom-ready reports.

---

## 📂 File Structure

The application is split into two clean files:
* `app.py`: **The Frontend.** This builds the beautiful Gradio user interface, handles the microphone input, renders the charts/maps, and manages the chat and audio interfaces.
* `village_engine.py`: **The Brains.** This handles all the heavy lifting. It talks to the AI models (Gemma), fetches data from satellites, runs the multi-agent debates, formats the PDFs, and calculates composite scores.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* **Python 3.10+** installed on your system.
* **Ollama:** You need to have [Ollama](https://ollama.com/) installed to run the local LLM. 
  * Open your terminal and run: `ollama run gemma:2b` (or whichever model you prefer) to download the AI brain.
* **Google Earth Engine (Optional but highly recommended):** For satellite data, you need an Earth Engine account. 

### 2. Install Python Dependencies
Open your terminal/command prompt, navigate to your project folder, and run the following command to install the required libraries:

```bash
pip install gradio earthengine-api plotly folium fpdf diskcache openai-whisper pyttsx3 gTTS requests python-dotenv
```

### 3. Fix Dependency Conflicts (Crucial Step)
Gradio relies on specific background libraries (`click` and `typer`). To prevent the app from crashing on startup with a `TypeError: type 'Choice' is not subscriptable` error, run this upgrade command:

```bash
pip install --upgrade click typer
```

### 4. Authenticate Earth Engine (If using satellite features)
Run this command in your terminal and follow the instructions in your browser to link your Google account:
```bash
earthengine authenticate
```

---

## 🚀 Running the Application

Once everything is installed, simply run:

```bash
python app.py
```

* You will see a link appear in your terminal (usually `http://127.0.0.1:7860` or `http://0.0.0.0:7860`). 
* `CTRL + Click` that link to open VillageForge AI in your web browser!

---

## 💡 Advanced Options (v3 Features)
When filling out the text form, you can open the **⚙️ Advanced Options** accordion to toggle:
* **Run Multi-Agent Debate:** Let the AI personas critique the plan (takes slightly longer, but yields much safer plans).
* **Generate Satellite Timelapse:** Creates a 5-year historical GIF of the site (adds ~30-60 seconds to processing time).
* **Beneficiary Personas & Tool Selection:** Uses the LLM to generate realistic human personas and dynamically choose which APIs to query based on your project type.

---

## 🛡️ Defensive Programming & Open Data
VillageForge AI is built to be robust. 
* **No Paid APIs:** It strictly uses open/free public data sources.
* **Graceful Degradation:** If Earth Engine is down, or a weather API fails, the app doesn't crash. It represents missing data as "Unknown" rather than faking a zero, and adapts the plan accordingly (e.g., "Flood risk data unavailable, defaulting to a conservative raised foundation"). 
```
