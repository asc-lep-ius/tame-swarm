"""
TAME Cortex Chat UI

Lightweight Gradio interface for testing the TAME architecture.
Connects to the local FastAPI server.

Usage:
    1. Start the main server: uvicorn main:app --host 0.0.0.0 --port 8000
    2. Run this: python chat_ui.py
    3. Open http://localhost:7860 in browser
"""

import gradio as gr
import requests
import os
from typing import List, Tuple

# API Configuration - use env var for Docker, default to localhost for local dev
API_BASE = os.getenv("TAME_API_URL", "http://localhost:8000")


def check_server_health() -> str:
    """Check if the TAME server is running."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.ok:
            data = resp.json()
            return f"✓ Connected | GPU: {data['gpu']} | MoB: {'✓' if data['mob_active'] else '✗'} | Steering: {'✓' if data['steering_active'] else '✗'}"
        return "✗ Server responded with error"
    except requests.exceptions.ConnectionError:
        return "✗ Cannot connect to server. Is it running on port 8000?"
    except Exception as e:
        return f"✗ Error: {e}"


def get_swarm_status() -> str:
    """Get current expert swarm status."""
    try:
        resp = requests.get(f"{API_BASE}/swarm/status", timeout=5)
        if resp.ok:
            data = resp.json()
            wealth = data["expert_wealth"]
            usage = data["expert_usage"]
            
            lines = [f"**Expert Swarm Status** ({data['num_experts']} experts, {data['layers_modified']} layers)"]
            lines.append("| Expert | Wealth | Usage |")
            lines.append("|--------|--------|-------|")
            for i, (w, u) in enumerate(zip(wealth, usage)):
                lines.append(f"| {i} | {w:.2f} | {int(u)} |")
            return "\n".join(lines)
        return "Failed to get swarm status"
    except Exception as e:
        return f"Error: {e}"


def get_homeostasis_status() -> str:
    """Get current homeostatic alignment."""
    try:
        resp = requests.get(f"{API_BASE}/homeostasis/status", timeout=5)
        if resp.ok:
            data = resp.json()
            if data["status"] == "disabled":
                return "Steering disabled"
            
            config = data["config"]
            stats = data.get("current_stats", {})
            
            lines = [
                "**Homeostasis Status**",
                f"- Base Strength: {config['base_strength']}",
                f"- Adaptive: {config['adaptive']}",
                f"- Target Alignment: {config['target_alignment']}",
            ]
            
            if stats:
                lines.append(f"- Current Alignment: {stats.get('mean_alignment', 'N/A')}")
                lines.append(f"- Steering Applied: {stats.get('steering_applied', 'N/A')}")
            
            return "\n".join(lines)
        return "Failed to get homeostasis status"
    except Exception as e:
        return f"Error: {e}"


def chat(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    max_tokens: int,
    steering_strength: float,
    show_stats: bool
) -> Tuple[str, str]:
    """Send message to TAME server and get response."""
    
    if not message.strip():
        return "", ""
    
    try:
        payload = {
            "prompt": message,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "return_stats": show_stats
        }
        
        # Only include steering_strength if not using adaptive (default)
        if steering_strength >= 0:
            payload["steering_strength"] = steering_strength
        
        resp = requests.post(
            f"{API_BASE}/generate",
            json=payload,
            timeout=120
        )
        
        if not resp.ok:
            return f"Error: {resp.status_code} - {resp.text}", ""
        
        data = resp.json()
        response_text = data["response"]
        
        # Build stats display
        stats_parts = []
        
        usage = data.get("usage", {})
        if usage:
            stats_parts.append(f"Tokens: {usage.get('input_tokens', '?')} → {usage.get('output_tokens', '?')}")
        
        homeostasis = data.get("homeostasis")
        if homeostasis:
            align = homeostasis.get("mean_alignment", "N/A")
            if isinstance(align, float):
                align = f"{align:.3f}"
            stats_parts.append(f"Alignment: {align}")
        
        mob = data.get("mob_stats")
        if mob:
            wealth = mob.get("expert_wealth", [])
            if wealth:
                top_expert = wealth.index(max(wealth))
                stats_parts.append(f"Top Expert: {top_expert} ({max(wealth):.1f})")
        
        stats_display = " | ".join(stats_parts) if stats_parts else ""
        
        return response_text, stats_display
        
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to server. Is it running?", ""
    except requests.exceptions.Timeout:
        return "Error: Request timed out", ""
    except Exception as e:
        return f"Error: {e}", ""


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="TAME Cortex Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 TAME Cortex Chat")
        gr.Markdown("Test the agential swarm with Mixture of Bidders and Cognitive Homeostasis")
        
        # Server status
        with gr.Row():
            status_box = gr.Textbox(
                label="Server Status",
                value=check_server_health(),
                interactive=False,
                scale=4
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)
        
        refresh_btn.click(check_server_health, outputs=status_box)
        
        with gr.Row():
            # Main chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                stats_display = gr.Textbox(
                    label="Generation Stats",
                    interactive=False,
                    visible=True
                )
                
                clear_btn = gr.Button("Clear Chat")
            
            # Settings column
            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=200,
                    step=10,
                    label="Max Tokens"
                )
                
                steering_strength = gr.Slider(
                    minimum=-0.1,
                    maximum=1.5,
                    value=-0.1,
                    step=0.1,
                    label="Steering Strength (-0.1 = adaptive)"
                )
                
                show_stats = gr.Checkbox(
                    value=True,
                    label="Show MoB Stats"
                )
                
                gr.Markdown("---")
                gr.Markdown("### System Status")
                
                swarm_btn = gr.Button("View Swarm Status")
                swarm_display = gr.Markdown("")
                
                homeo_btn = gr.Button("View Homeostasis")
                homeo_display = gr.Markdown("")
        
        # Event handlers
        def respond(message, history, temp, max_tok, steer, stats):
            response, stats_text = chat(message, history, temp, max_tok, steer, stats)
            history = history + [(message, response)]
            return "", history, stats_text
        
        msg_input.submit(
            respond,
            [msg_input, chatbot, temperature, max_tokens, steering_strength, show_stats],
            [msg_input, chatbot, stats_display]
        )
        
        send_btn.click(
            respond,
            [msg_input, chatbot, temperature, max_tokens, steering_strength, show_stats],
            [msg_input, chatbot, stats_display]
        )
        
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, stats_display])
        
        swarm_btn.click(get_swarm_status, outputs=swarm_display)
        homeo_btn.click(get_homeostasis_status, outputs=homeo_display)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
