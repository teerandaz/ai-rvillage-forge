***

# 🏗️ VillageForge AI v3
### *Agentic, Open-Data Infrastructure Planner for Rural Resilience*

**VillageForge AI** is a multimodal tool designed to help NGOs, governments, and engineers plan rural infrastructure (schools, clinics, solar grids) using nothing but open-access data. It replaces expensive site surveys with a high-speed, AI-driven analysis of satellite imagery, climate history, and social data.

---

## 🌟 How it Works (The Simple Version)
Think of VillageForge as a **Digital Architect and Engineer** in your pocket. 

1.  **Search:** You give it a location (like a village name or pincode) and a budget.
2.  **Analyze:** The app automatically scans dozens of "Open Data" layers—it checks if the ground is prone to flooding, how much sun hits the roof for solar panels, where the nearest hospital is, and even what the climate will look like in the year 2050.
3.  **Plan:** A Large Language Model (Gemma) acts as a project manager to write a full construction plan, including a budget breakdown and a timeline.
4.  **Debate:** To ensure safety, three AI "Agents" (a Structural Engineer, a Climate Scientist, and a Social Advocate) look at the plan and argue about how to make it better.
5.  **Output:** You get a professional PDF report, a 3D interactive map, and a 5-year project roadmap.

---

## 🚀 Getting Started

### 1. Prerequisites
*   **Python 3.9+**
*   **A GPU** (NVIDIA T4 or better recommended) if running locally.
*   **Google Earth Engine Account** (Optional but recommended for maps).

### 2. Installation
Run the following commands in your terminal or Kaggle cell to install the engine and its interface:

```bash
# Upgrade core CLI tools to avoid dependency conflicts
pip install --upgrade click typer

# Install the UI and Data Visualization libraries
pip install gradio plotly folium fpdf requests

# Install the AI and Satellite Processing libraries
pip install torch transformers bitsandbytes accelerate
pip install earthengine-api diskcache python-dotenv

# Install the Audio/Voice processing libraries
pip install openai-whisper pyttsx3
```

### 3. Setup
1.  **Model Path:** Ensure your Gemma model path is correctly set in `village_engine.py`. 
2.  **Environment:** If you have a `.env` file, place your API keys there (e.g., `NASA_FIRMS_MAP_KEY`).

---

## 🛠️ Usage

To start the VillageForge interface, run:

```bash
python app.py
```

If you are using **Kaggle**, run this in a cell:
```python
!python /kaggle/working/ai-remote-builder/app.py
```
*Wait for the `Public URL` (ending in `.gradio.live`) to appear and click it.*

---

## 📂 Project Structure

| File | Role |
| :--- | :--- |
| `app.py` | **The Face:** The Gradio web interface, charts, and interactive 3D maps. |
| `village_engine.py` | **The Brain:** Handles geocoding, satellite data fetching, AI agent logic, and PDF generation. |

---

## 📊 Key Functions Explained

*   **`run_village_analysis`**: The main orchestrator. It triggers the geocoding, calls the data tools, and generates the AI plan.
*   **`analyze_flood_risk` / `analyze_seismic_risk`**: These functions use NASA and USGS data to determine if the site is safe for building.
*   **`run_agent_debate`**: This is where the "Agentic" part happens. It forces the AI to look at its own plan from different professional perspectives to find errors.
*   **`generate_3d_visualization`**: Uses `deck.gl` to turn flat risk scores into a 3D landscape you can rotate and zoom.
*   **`compute_composite_scores`**: A mathematical formula that combines all data into simple 0-100 scores for "Construction Suitability" or "Water Security."

---

## 📡 Data Sources Used
VillageForge is "Open Data First," meaning it uses free public information:
*   **Satellite:** Sentinel-2, Landsat (via Google Earth Engine)
*   **Environment:** NASA POWER (Solar/Wind), CHIRPS (Rainfall)
*   **Infrastructure:** OpenStreetMap (Roads, Hospitals, Buildings)
*   **Social:** WHO & World Bank (Health and Economic indicators)
*   **Risk:** USGS (Earthquakes), JRC (Flood/Water recurrence)

---

## ⚖️ License & Disclaimer
*VillageForge AI is a decision-support tool. It is intended for preliminary planning and should always be followed by a physical on-site survey by qualified professionals.*

---
*Created for the Social Impact AI Hackathon.*
