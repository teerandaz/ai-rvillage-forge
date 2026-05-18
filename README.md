***

# 🏗️ VillageForge AI v3
### *Agentic, Multimodal, Open-Data Infrastructure Planner for Rural Resilience*

**VillageForge AI** is a complete, agentic infrastructure planning suite. It enables anyone—from field workers to government officials—to generate expert-level, climate-resilient construction plans for schools, clinics, and utilities using only open-source data.

---

## 🎙️ Multimodal & Multilingual Voice Features
One of the most powerful features of VillageForge is its **Multimodal Interface**, designed for accessibility in areas where typing or English proficiency might be a barrier.

*   **Multilingual Voice Input (Audio-In):**
    *   **How it works:** Uses the **OpenAI Whisper** model locally.
    *   **Translation:** You can speak in your native language (e.g., Hindi, Swahili, Spanish, etc.), and the system automatically transcribes and translates your intent into English planning instructions.
    *   **Intent Extraction:** The AI extracts the **Location**, **Project Type**, and **Budget** directly from your spoken words.
*   **Local Voice Output (Audio-Out):**
    *   **How it works:** Uses a local, offline **Text-to-Speech (TTS)** engine (`pyttsx3`).
    *   **Accessibility:** Once a plan is generated, the AI creates a spoken summary of the project. This allows non-technical stakeholders to "hear" the project viability and risks without reading a 20-page report.
    *   **Zero-Cloud:** The voice generation happens entirely on your machine—no data is sent to private cloud TTS providers.

---

## 🧠 The "Agentic" Planning Engine
VillageForge doesn't just "guess"; it simulates a professional boardroom through a **Multi-Agent Debate**:

1.  **The Architect (Lead):** Drafts the initial structure and budget.
2.  **The Structural Engineer:** Critiques the plan for earthquake, flood, and soil bearing safety.
3.  **The Climate Scientist:** Stress-tests the design against the year 2050 heatwaves and chronic drought.
4.  **The Social Impact Advocate:** Ensures the design is inclusive for women, children, and the elderly.
5.  **The Budget Auditor:** Strips away waste and ensures the project is financially realistic.

**Result:** The final plan is a "Consensus" document that has been cross-checked by 5 different AI perspectives.

---

## 🚀 Installation & Command Guide

Run these commands to ensure every feature (maps, 3D, PDF, Voice, AI) works perfectly.

### 1. Core System Upgrade
```bash
# Vital to avoid dependency conflicts with Gradio and Typer
pip install --upgrade click typer
```

### 2. Full Feature Installation
```bash
# UI and Visuals
pip install gradio plotly folium fpdf requests

# AI Brain & Local Model Support
pip install torch transformers bitsandbytes accelerate

# Data Science & Satellite Processing
pip install earthengine-api diskcache python-dotenv math re

# Multilingual Voice (Audio-In & Audio-Out)
pip install openai-whisper pyttsx3
```

---

## 🛠️ Feature Breakdown

### 📊 Professional Planning Tools
*   **Budget & Timeline:** Automatic breakdown of costs with a "Gantt-style" timeline of construction phases.
*   **SDG Impact Calculator:** Specifically maps the project to **United Nations Sustainable Development Goals** and calculates the "Cost per Beneficiary."
*   **Beneficiary Personas:** Generates realistic, data-driven "Human Profiles" of people who will use the facility (e.g., a local teacher or a mother of three).
*   **Plan Diffing:** If you change a location or budget, the AI runs a "Semantic Diff" to show you exactly how the strategy changed between Version A and Version B.

### 🌍 Satellite & Environmental Intelligence
*   **3D Site Visualization:** A high-tech `deck.gl` 3D view that turns flood and seismic risks into a "heat-map" landscape.
*   **Sentinel-2 Timelapse:** (Requires Earth Engine) Watch 5 years of historical land changes at the site to check for erosion or urban growth.
*   **Live Site Alerts:** Connects to Open-Meteo and NASA FIRMS to give you real-time weather and fire alerts for the chosen coordinates.
*   **2050 Climate Stress Test:** Modeled projections of how much hotter/drier the site will be in 30 years.

### 📄 Exportable Artifacts
*   **PDF Reports:** A fully formatted, multi-page professional engineering report ready for printing.
*   **Interactive HTML Maps:** Folium-based maps showing nearby hospitals, schools, and roads.
*   **JSON Data:** Export the raw data to integrate with other government GIS systems.

---

## 📂 How to Run

### On Local Machine:
1.  Open your terminal.
2.  Run `python app.py`.
3.  Open `http://localhost:7860`.

### On Kaggle:
1.  Enable **GPU T4 x2** in Settings.
2.  Paste the code into a cell.
3.  Run the cell and wait for the **Gradio Public URL** (`xxxx.gradio.live`).
4.  **Note:** The first run takes ~2 minutes to load the 2B/4B Gemma model into the GPU memory.

---

## 📡 The Open Data "Sources"
VillageForge is powered by the world's most reliable open data:
*   **Infrastructure:** OpenStreetMap (OSM)
*   **Climate/Weather:** NASA POWER & Open-Meteo
*   **Health:** WHO Global Health Observatory
*   **Economy:** World Bank API
*   **Agriculture:** FAOSTAT
*   **Disasters:** ReliefWeb & USGS
*   **Satellite Imagery:** European Space Agency (ESA) & Copernicus

---

## 📛 Why "VillageForge AI"?
**Village** represents our focus: the underserved, rural, and last-mile communities.  
**Forge** represents the process: high-pressure, durable, and expert creation.  
**AI** represents the engine: agentic, multimodal, and data-driven.

---
*Created for the Social Impact AI Hackathon. Empowering the next billion through data-driven resilience.*
