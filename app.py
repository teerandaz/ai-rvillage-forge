"""
VillageForge AI v3 — Multimodal & Agentic Social Good Planning Agent
Run: python app.py
Opens at http://localhost:7860
"""

import gradio as gr
import ee
import json
import os
import sys
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# Import gTTS for multilingual audio support
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
    print("[Warning] 'gTTS' is not installed. Please run 'pip install gTTS' for multilingual audio.")

sys.path.insert(0, os.path.dirname(__file__))

# Import the v3 functions from the engine
from village_engine import (
    run_village_analysis,
    process_voice_audio,
    speak_plan_summary,
    chat_with_plan,
    call_gemma,
    sanitize_text,
    GEMMA_MODEL,
    EE_PROJECT,
)

def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT)
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=EE_PROJECT)
            return True
        except Exception as e:
            print(f"[EE] Warning: {e}")
            return False

EE_OK = init_ee()

# Multilingual mapping for TTS
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Swahili": "sw",
    "Arabic": "ar"
}


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def build_budget_chart(plan: dict) -> go.Figure:
    budget = plan.get("budget_allocation", {}).get("breakdown", {})
    labels, values, notes = [], [], []
    for phase, alloc in budget.items():
        if isinstance(alloc, dict) and alloc.get("amount_usd", 0) > 0:
            labels.append(str(phase).title())
            values.append(alloc["amount_usd"])
            notes.append(alloc.get("notes", ""))
    if not labels: return go.Figure()
    fig = px.pie(names=labels, values=values, title="Budget Allocation", color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
    fig.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
    fig.update_layout(showlegend=True, margin=dict(t=40, b=0, l=0, r=0), height=360)
    return fig

def build_scores_chart(composite_scores: dict) -> go.Figure:
    score_keys = [k for k, v in composite_scores.items() if isinstance(v, (int, float))]
    labels = [k.replace("_", " ").title() for k in score_keys]
    values = [composite_scores[k] for k in score_keys]
    colors = ["#4CAF50" if v > 60 else "#FF9800" if v > 35 else "#F44336" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=[f"{v:.0f}/100" for v in values], textposition="outside"))
    fig.update_layout(title="Composite Site Scores", xaxis=dict(range=[0, 110], showgrid=False), yaxis=dict(autorange="reversed"), margin=dict(t=40, b=0, l=0, r=60), height=240, plot_bgcolor="rgba(0,0,0,0)")
    return fig

def build_timeline_chart(plan: dict) -> go.Figure:
    sectors = plan.get("sectors", {})
    if not sectors: return go.Figure()
    start = 0
    fig = go.Figure()
    for i, (name, detail) in enumerate(sectors.items()):
        if not isinstance(detail, dict): continue
        w = detail.get("timeline_weeks", 4)
        c = detail.get("estimated_cost_usd", 0)
        fig.add_trace(go.Bar(name=str(name).title(), x=[w], y=[str(name).title()[:30]], base=[start], orientation="h", hovertemplate=f"<b>{str(name).title()}</b><br>Duration: {w} weeks<br>Cost: ${c:,}<extra></extra>"))
        start += w
    fig.update_layout(barmode="stack", title="Construction Timeline (Gantt)", xaxis_title="Weeks from project start", showlegend=False, margin=dict(t=40, b=30, l=180, r=20), height=max(200, len(sectors) * 40 + 80), plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ─────────────────────────────────────────────
# MAIN HANDLERS
# ─────────────────────────────────────────────
def analyse_village(location_str: str, project_type: str, budget: int, run_debate: bool, gen_timelapse: bool, use_personas: bool, use_tools: bool, progress=gr.Progress()):
    empty_returns = (gr.update(value="❌ Please enter a pincode or city name."),) + (None,) * 12
    if not location_str.strip():
        yield empty_returns
        return

    steps = [
        "Geocoding location...", 
        "Phase 1: Gemma selecting relevant analysis tools...", 
        "Phase 2: Running satellite layers in parallel (GEE)...", 
        "Fetching external APIs...", 
        "Computing composite site scores...", 
        "Phase 3: Gemma architecting the project plan...", 
        "Generating interactive site map & 3D models...", 
        "Compiling PDF report & artifacts..."
    ]
    for i, step in enumerate(steps): progress((i + 1) / len(steps), desc=step)

    try: 
        result = run_village_analysis(
            location_str.strip(), 
            int(budget), 
            project_type,
            run_debate=run_debate,
            generate_timelapse=gen_timelapse,
            use_llm_personas=use_personas,
            use_llm_tool_selection=use_tools
        )
    except Exception as e:
        import traceback
        error_resp = (gr.update(value=f"❌ Error: {e}\n\n```\n{traceback.format_exc()}\n```"),) + (None,) * 12
        yield error_resp
        return

    plan, raw, loc, cs = result["plan"], result["raw_data"], result["location"], result["composite_scores"]
    pdf_path = result.get("pdf_path")
    json_path = result.get("json_path")
    map_path = result.get("map_path")
    view_3d_path = result.get("view_3d_path")
    timelapse_path = result.get("timelapse_path")

    if plan.get("parse_error"):
        status_md = f"⚠️ Plan generated but JSON parsing failed.\n\n```\n{plan.get('raw_response','')[:600]}\n```"
    else:
        warnings = plan.get("warnings", [])
        warn_str = "\n".join(f"- ⚠️ {w}" for w in warnings) if warnings else "- ✅ No critical site risks detected"
        priority_str = " → ".join(str(p).title() for p in plan.get("priority_ranking", [])) or "N/A"
        total = plan.get("budget_allocation", {}).get("total_recommended_usd", 0)
        viability = plan.get("site_viability", "N/A").upper()
        viability_icon = "✅" if viability == "VIABLE" else "⚠️" if viability == "MARGINAL" else "❌"

        status_md = f"""## {viability_icon} {project_type} Plan — {loc.get('district', '')}, {loc.get('state', '')}
**Site Viability:** {viability_icon} {viability}
**Summary:** {plan.get('project_summary', '')}
**Construction Suitability:** {cs.get('construction_suitability', 0):.0f}/100 ({cs.get('construction_suitability_label','')})
**Renewable Energy:** {cs.get('renewable_energy_score', 0):.0f}/100 (primary: {cs.get('primary_renewable','')})
**Water Security:** {cs.get('water_security_score', 0):.0f}/100 ({cs.get('water_security_label','')})
**Accessibility:** {cs.get('accessibility_score', 0):.0f}/100 ({cs.get('accessibility_label','')})

**Action Timeline:** {priority_str}
**Estimated Project Cost:** ${total:,}

**Critical Site Risks:**
{warn_str}

---
**Data layers analysed by Gemma:** `{'`, `'.join(result['selected_tools'])}`
"""
    raw_md = f"```json\n{json.dumps(raw, indent=2)}\n```"

    sectors_md = ""
    for phase, detail in plan.get("sectors", {}).items():
        if not isinstance(detail, dict): continue
        item_str = "\n".join(f"  - {i}" for i in detail.get("items", [])) if detail.get("items") else "  - See PDF for details"
        sectors_md += f"### 🏗 {str(phase).title()}\n**Cost:** ${detail.get('estimated_cost_usd', 0):,} | **Timeline:** {detail.get('timeline_weeks', '?')} weeks\n**Rationale:** {detail.get('rationale', '')}\n**Materials:**\n{item_str}\n\n"

    yield (
        status_md + "\n---\n" + sectors_md, 
        pdf_path, raw_md, pdf_path, json_path, 
        build_budget_chart(plan), build_scores_chart(cs), build_timeline_chart(plan), 
        map_path, view_3d_path, timelapse_path, plan, raw
    )


def generate_multilingual_audio(plan: dict, target_lang_name: str):
    """Translates the summary using Gemma and reads it out using gTTS"""
    if not plan: return None
    
    text = sanitize_text(plan.get("project_summary", ""))[:900]
    if not text: return None
    
    lang_code = LANGUAGES.get(target_lang_name, "en")
    
    # 1. Ask Gemma to translate if the target is not English
    if lang_code != "en":
        try:
            print(f"[Translation] Asking Gemma to translate summary to {target_lang_name}...")
            prompt = f"Translate the following text into {target_lang_name}. Output ONLY the translated text without any conversational fillers or quotes.\n\nText: {text}"
            text = call_gemma(prompt, system="You are a professional, direct translator.", timeout=60).strip()
        except Exception as e:
            print(f"[Translation Error] Failed to reach Gemma: {e}")
            lang_code = "en" # Fallback to English if LLM fails
            
    # 2. Synthesize Audio
    if gTTS is None:
        print("[TTS Warning] gTTS not installed. Falling back to default offline pyttsx3 (English only).")
        return speak_plan_summary(plan)
        
    try:
        print(f"[TTS] Generating audio in {lang_code}...")
        tts = gTTS(text=text, lang=lang_code)
        
        # Create temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.close()
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
PROJECT_OPTIONS = ["General Area Analysis", "Primary School", "Rural Health Clinic / Hospital", "Solar Microgrid Utility", "Water Purification & Storage Center", "Community Farming Hub", "Disaster Relief Shelter"]
EXAMPLES = [["843302", "Rural Health Clinic / Hospital", 80000], ["Mumbai", "Primary School", 150000]]

# Note: Removed the `theme` from blocks to fix the warning. (It's now added in `.launch()`)
with gr.Blocks(title="VillageForge AI") as demo:
    
    plan_state = gr.State()
    raw_data_state = gr.State()

    with gr.Column(visible=True) as screen_intro:
        gr.Markdown("# 🏗️ VillageForge AI — Multimodal Infrastructure Planner")
        gr.Markdown("""### Select how you would like to begin your planning:""")
        with gr.Row():
            btn_go_voice = gr.Button("🎤 Proceed with Voice (Multilingual)", variant="primary", size="lg")
            btn_go_text  = gr.Button("⌨️ Proceed with Text Form", size="lg")

    with gr.Column(visible=False) as screen_voice:
        gr.Markdown("## 🎤 Multilingual Voice Assistant")
        gr.Markdown("*Speak in **any language**! Mention the **location** (city or pincode), what you want to **build**, and your **budget**.*")
        
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record your request")
        
        with gr.Row():
            btn_back_voice = gr.Button("⬅️ Back to Start")
            btn_process_voice = gr.Button("🔄 Translate & Extract Intent", variant="primary")
            
        voice_status = gr.Textbox(label="Transcription & Status", interactive=False)

    with gr.Column(visible=False) as screen_main:
        with gr.Row():
            btn_back_main = gr.Button("⬅️ Back to Start", size="sm")
            gr.Markdown("## ⌨️ Verify & Generate Project Plan")
            
        with gr.Row():
            with gr.Column(scale=1):
                location_in = gr.Textbox(label="Location (City or Pincode)", placeholder="e.g. 843302 or Mumbai")
                project_in  = gr.Dropdown(choices=PROJECT_OPTIONS, value="General Area Analysis", label="What do you want to build?")
                budget_in   = gr.Slider(label="Maximum Budget Limit (USD)", minimum=5000, maximum=500000, value=50000, step=5000)
                
                with gr.Accordion("⚙️ Advanced Options (v3 Features)", open=False):
                    chk_debate = gr.Checkbox(label="Run Multi-Agent Debate", value=True)
                    chk_timelapse = gr.Checkbox(label="Generate Satellite Timelapse (GIF)", value=False)
                    chk_personas = gr.Checkbox(label="Generate Beneficiary Personas", value=True)
                    chk_tools = gr.Checkbox(label="Use LLM for Tool Selection", value=True)

                analyse_btn = gr.Button("🔍 Generate Data-Driven Plan", variant="primary", size="lg")
                gr.Examples(examples=EXAMPLES, inputs=[location_in, project_in, budget_in], label="Example Projects")

            with gr.Column(scale=2):
                output_md   = gr.Markdown(label="Development Plan")
                pdf_preview = gr.File(label="📄 Download PDF Report", visible=True)

        with gr.Tabs():
            with gr.TabItem("📊 Charts"):
                with gr.Row():
                    scores_chart  = gr.Plot(label="Composite Site Scores")
                    budget_chart  = gr.Plot(label="Budget Allocation")
                timeline_chart = gr.Plot(label="Construction Timeline")

            with gr.TabItem("🗺 2D Site Map"):
                map_file = gr.File(label="Download 2D site map HTML")

            with gr.TabItem("🌍 3D Risk Map"):
                view_3d_file = gr.File(label="Download interactive 3D View HTML (Open in Browser)")

            with gr.TabItem("🛰️ Satellite Timelapse"):
                gr.Markdown("*(Ensure you enabled 'Generate Satellite Timelapse' in Advanced Options before running)*")
                timelapse_image = gr.Image(label="5-Year Sentinel-2 Timelapse", type="filepath")

            with gr.TabItem("💬 Chat & Audio"):
                gr.Markdown("### Chat directly with your Infrastructure Plan")
                # Removed type="messages" because Gradio 5+ defaults to dictionary format and rejects the keyword.
                chatbot = gr.Chatbot(height=300) 
                msg = gr.Textbox(label="Ask a question about the generated plan (e.g., 'What is the flood risk?')")
                
                def respond(message, chat_history, plan, raw):
                    if chat_history is None:
                        chat_history = []
                        
                    bot_message = chat_with_plan(message, chat_history, plan, raw)
                    
                    # Appending natively as dictionaries as required by Gradio 5+
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": bot_message})
                    return "", chat_history
                
                msg.submit(respond, [msg, chatbot, plan_state, raw_data_state], [msg, chatbot])

                gr.Markdown("---")
                gr.Markdown("### 🗣️ Multilingual Audio Summary")
                
                with gr.Row():
                    lang_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="English", label="Select Output Language")
                    btn_audio = gr.Button("Generate Voice Summary", variant="secondary")
                
                audio_out = gr.Audio(label="Audio Output", autoplay=True)
                
                btn_audio.click(generate_multilingual_audio, inputs=[plan_state, lang_dropdown], outputs=[audio_out])

            with gr.TabItem("📊 Raw Site Data"):
                raw_json_out = gr.Markdown()

            with gr.TabItem("📁 Download Files"):
                pdf_dl  = gr.File(label="PDF Report")
                json_dl = gr.File(label="JSON Data")

    # --- NAVIGATION LOGIC ---
    def show_voice(): return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    def show_text(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    def show_intro(): return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    btn_go_voice.click(show_voice, None, [screen_intro, screen_voice, screen_main])
    btn_go_text.click(show_text, None, [screen_intro, screen_voice, screen_main])
    btn_back_voice.click(show_intro, None, [screen_intro, screen_voice, screen_main])
    btn_back_main.click(show_intro, None, [screen_intro, screen_voice, screen_main])

    # --- VOICE LOGIC ---
    def handle_voice_recording(audio_path):
        if not audio_path:
            yield "❌ Please record audio first.", gr.update(), gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)
            return
            
        yield "⏳ Transcribing and translating (via Whisper)...", gr.update(), gr.update(), gr.update(), gr.update(visible=True), gr.update(visible=False)
        
        result = process_voice_audio(audio_path)
        msg = f"✅ Translated: '{result.get('english_text')}'\n🧠 Intent -> Location: {result.get('location')}, Project: {result.get('project_type')}, Budget: ${result.get('budget')}"
        
        yield msg, result.get("location", ""), result.get("project_type", "General Area Analysis"), result.get("budget", 50000), gr.update(visible=False), gr.update(visible=True)

    btn_process_voice.click(
        handle_voice_recording,
        inputs=[audio_in],
        outputs=[voice_status, location_in, project_in, budget_in, screen_voice, screen_main]
    )

    analyse_btn.click(
        analyse_village,
        inputs=[location_in, project_in, budget_in, chk_debate, chk_timelapse, chk_personas, chk_tools],
        outputs=[
            output_md, pdf_preview, raw_json_out, pdf_dl, json_dl, 
            budget_chart, scores_chart, timeline_chart, map_file,
            view_3d_file, timelapse_image, plan_state, raw_data_state
        ]
    )

if __name__ == "__main__":
    demo.queue()
    # Pushing the "theme" config into launch() fixes the Gradio 6.0 warning
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, theme=gr.themes.Soft())
