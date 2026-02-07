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
import json
import os
from typing import List, Tuple, Generator, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# API Configuration - use env var for Docker, default to localhost for local dev
API_BASE = os.getenv("TAME_API_URL", "http://localhost:8000")


def create_wealth_distribution_plot(wealth_trace: Dict[str, Any]) -> go.Figure:
    """
    Create a plotly figure showing expert wealth distribution over tokens.
    
    Good Sign: Inequality - some experts should become rich (specialists),
    others poor. Equal wealth means the auction isn't forcing specialization.
    """
    if not wealth_trace or not wealth_trace.get("expert_wealth"):
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No wealth trace data available yet.<br>Generate a response to see VCG auction dynamics.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="VCG Auction: Expert Wealth Distribution",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=350
        )
        return fig
    
    expert_wealth = wealth_trace["expert_wealth"]
    steps = wealth_trace.get("steps", list(range(len(expert_wealth))))
    num_experts = wealth_trace.get("num_experts", len(expert_wealth[0]) if expert_wealth else 0)
    
    # Create traces for each expert
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for expert_idx in range(num_experts):
        wealth_values = [step[expert_idx] for step in expert_wealth]
        fig.add_trace(go.Scatter(
            x=steps,
            y=wealth_values,
            mode='lines',
            name=f'Expert {expert_idx}',
            line=dict(width=2, color=colors[expert_idx % len(colors)]),
            hovertemplate=f'Expert {expert_idx}<br>Step: %{{x}}<br>Wealth: %{{y:.2f}}<extra></extra>'
        ))
    
    # Calculate inequality metric (Gini coefficient approximation)
    if expert_wealth:
        final_wealth = expert_wealth[-1]
        sorted_wealth = sorted(final_wealth)
        n = len(sorted_wealth)
        if n > 0 and sum(sorted_wealth) > 0:
            cum_wealth = sum((i + 1) * w for i, w in enumerate(sorted_wealth))
            total_wealth = sum(sorted_wealth)
            gini = (2 * cum_wealth) / (n * total_wealth) - (n + 1) / n
        else:
            gini = 0
        inequality_text = f"Gini: {gini:.3f} ({'Specializing!' if gini > 0.2 else 'Low inequality'})"
    else:
        inequality_text = ""
    
    fig.update_layout(
        title=f"VCG Auction: Expert Wealth Over Tokens<br><sub>{inequality_text}</sub>",
        xaxis_title="Forward Pass (≈ tokens)",
        yaxis_title="Wealth (credits)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=50, r=20, t=80, b=50),
        hovermode='x unified'
    )
    
    return fig


def create_steering_trace_plot(steering_trace: Dict[str, Any]) -> go.Figure:
    """
    Create a plotly figure showing steering strength (α_t) over time.
    
    Good Sign: Dynamic behavior - low when naturally aligned with goal,
    spikes when the prompt tries to force the model off-course.
    """
    if not steering_trace or not steering_trace.get("strength_history"):
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No steering trace data available yet.<br>Generate a response to see homeostatic dynamics.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Homeostatic Trace: Steering Strength (α_t)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=350
        )
        return fig
    
    strength_history = steering_trace["strength_history"]
    alignment_history = steering_trace.get("alignment_history", [])
    target_alignment = steering_trace.get("target_alignment", 0.7)
    steps = list(range(len(strength_history)))
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Steering strength trace
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=strength_history,
            mode='lines',
            name='Steering Strength (α_t)',
            line=dict(width=2.5, color='#FF6B6B'),
            hovertemplate='Step: %{x}<br>Strength: %{y:.3f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Alignment trace (if available)
    if alignment_history:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(alignment_history))),
                y=alignment_history,
                mode='lines',
                name='Alignment',
                line=dict(width=2, color='#4ECDC4', dash='dot'),
                hovertemplate='Step: %{x}<br>Alignment: %{y:.3f}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Target alignment line
        fig.add_hline(
            y=target_alignment,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Target ({target_alignment})",
            secondary_y=True
        )
    
    # Analyze dynamics
    if strength_history:
        variance = sum((s - sum(strength_history)/len(strength_history))**2 for s in strength_history) / len(strength_history)
        dynamics_text = f"Variance: {variance:.4f} ({'Dynamic!' if variance > 0.01 else 'Static - may need tuning'})"
    else:
        dynamics_text = ""
    
    fig.update_layout(
        title=f"Homeostatic Trace: Injection Strength Over Time<br><sub>{dynamics_text}</sub>",
        xaxis_title="Forward Pass",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Strength (α_t)", secondary_y=False)
    fig.update_yaxes(title_text="Alignment (cosine sim)", secondary_y=True)
    
    return fig


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
            
            # Create list of (expert_id, wealth, usage) tuples and sort by usage descending
            experts = [(i, w, u) for i, (w, u) in enumerate(zip(wealth, usage))]
            experts.sort(key=lambda x: x[2], reverse=True)
            
            lines = [f"**Expert Swarm Status** ({data['num_experts']} experts, {data['layers_modified']} layers)"]
            lines.append("| Expert | Wealth | Usage |")
            lines.append("|--------|--------|-------|")
            for expert_id, w, u in experts:
                lines.append(f"| {expert_id} | {w:.2f} | {int(u)} |")
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


def stream_chat(
    message: str,
    history: List[dict],
    temperature: float,
    max_tokens: int,
    steering_strength: float,
    show_stats: bool
) -> Generator[Tuple[str, str, Dict[str, Any], Dict[str, Any]], None, None]:
    """
    Stream response from TAME server with real-time token feedback.
    
    Yields:
        Tuple of (accumulated_response, status_text, wealth_trace, steering_trace) for each token
    """
    if not message.strip():
        yield "", "", {}, {}
        return
    
    try:
        payload = {
            "prompt": message,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "return_stats": show_stats
        }
        
        if steering_strength >= 0:
            payload["steering_strength"] = steering_strength
        
        # Use streaming endpoint with much longer timeout (10 minutes)
        response = requests.post(
            f"{API_BASE}/generate/stream",
            json=payload,
            stream=True,
            timeout=600  # 10 minute timeout for long generations
        )
        
        if not response.ok:
            yield f"Error: {response.status_code} - {response.text}", "", {}, {}
            return
        
        accumulated_response = ""
        status_text = "Generating..."
        wealth_trace = {}
        steering_trace = {}
        
        # Process Server-Sent Events
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            if not line.startswith('data: '):
                continue
            
            data_str = line[6:]  # Remove 'data: ' prefix
            
            if data_str == '[DONE]':
                break
            
            try:
                data = json.loads(data_str)
                event_type = data.get('type', '')
                
                if event_type == 'token':
                    # Append token to response
                    accumulated_response += data.get('content', '')
                    yield accumulated_response, status_text, wealth_trace, steering_trace
                    
                elif event_type == 'status':
                    status_text = f"⏳ {data.get('message', '')}"
                    yield accumulated_response, status_text, wealth_trace, steering_trace
                    
                elif event_type == 'progress':
                    status_text = f"🔄 {data.get('message', '')} tokens"
                    yield accumulated_response, status_text, wealth_trace, steering_trace
                    
                elif event_type == 'complete':
                    # Extract traces from final stats
                    wealth_trace = data.get('wealth_trace', {})
                    steering_trace = data.get('steering_trace', {})
                    
                    # Build final stats display
                    stats_parts = []
                    
                    usage = data.get('usage', {})
                    if usage:
                        stats_parts.append(f"Tokens: {usage.get('input_tokens', '?')} → {usage.get('output_tokens', '?')}")
                    
                    homeostasis = data.get('homeostasis')
                    if homeostasis:
                        align = homeostasis.get('mean_alignment', 'N/A')
                        if isinstance(align, float):
                            align = f"{align:.3f}"
                        stats_parts.append(f"Alignment: {align}")
                    
                    mob = data.get('mob_stats')
                    if mob:
                        wealth = mob.get('expert_wealth', [])
                        if wealth:
                            top_expert = wealth.index(max(wealth))
                            stats_parts.append(f"Top Expert: {top_expert} ({max(wealth):.1f})")
                    
                    status_text = "✓ " + " | ".join(stats_parts) if stats_parts else "✓ Complete"
                    yield accumulated_response, status_text, wealth_trace, steering_trace
                    
                elif event_type == 'error':
                    yield f"Error: {data.get('message', 'Unknown error')}", "❌ Error", {}, {}
                    return
                    
            except json.JSONDecodeError:
                continue
        
        # Final yield
        yield accumulated_response, status_text, wealth_trace, steering_trace
        
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to server. Is it running?", "❌ Connection Error", {}, {}
    except requests.exceptions.Timeout:
        yield "Error: Request timed out (10 min limit)", "❌ Timeout", {}, {}
    except Exception as e:
        yield f"Error: {e}", "❌ Error", {}, {}


# Keep non-streaming version as fallback
def chat(
    message: str,
    history: List[dict],
    temperature: float,
    max_tokens: int,
    steering_strength: float,
    show_stats: bool
) -> Tuple[str, str]:
    """Send message to TAME server and get response (non-streaming fallback)."""
    
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
            timeout=600  # Increased to 10 minutes
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
    
    with gr.Blocks(title="TAME Cortex Chat") as demo:
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
                    height=350
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
        
        # TAME Visualization Section
        gr.Markdown("---")
        gr.Markdown("## 📊 TAME Architecture Insights")
        gr.Markdown("""
        **VCG Auction (MoB)**: Shows expert wealth distribution over tokens. 
        *Good sign: Inequality - some experts should become rich specialists!*
        
        **Homeostatic Trace**: Shows steering strength (α_t) over time.
        *Good sign: Dynamic - should spike when model drifts from goals!*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                wealth_plot = gr.Plot(
                    label="VCG Auction: Expert Wealth Distribution",
                    value=create_wealth_distribution_plot({})
                )
            with gr.Column(scale=1):
                steering_plot = gr.Plot(
                    label="Homeostatic Trace: Steering Strength",
                    value=create_steering_trace_plot({})
                )
        
        # Event handlers - streaming version with trace visualization
        def respond_stream(message, history, temp, max_tok, steer, stats):
            """Streaming response handler for Gradio with trace visualization."""
            if not message.strip():
                empty_wealth_plot = create_wealth_distribution_plot({})
                empty_steering_plot = create_steering_trace_plot({})
                yield "", history, "", empty_wealth_plot, empty_steering_plot
                return
            
            # Add user message to history immediately
            history = history + [{"role": "user", "content": message}]
            
            # Stream the response
            accumulated = ""
            status = "Starting..."
            wealth_trace = {}
            steering_trace = {}
            
            for response_text, stats_text, w_trace, s_trace in stream_chat(message, history, temp, max_tok, steer, stats):
                accumulated = response_text
                status = stats_text
                wealth_trace = w_trace if w_trace else wealth_trace
                steering_trace = s_trace if s_trace else steering_trace
                
                # Update history with current accumulated response
                current_history = history + [{"role": "assistant", "content": accumulated}]
                
                # Create plots (only update when we have data)
                wealth_fig = create_wealth_distribution_plot(wealth_trace)
                steering_fig = create_steering_trace_plot(steering_trace)
                
                yield "", current_history, status, wealth_fig, steering_fig
            
            # Final yield with complete response and final plots
            final_history = history + [{"role": "assistant", "content": accumulated}]
            wealth_fig = create_wealth_distribution_plot(wealth_trace)
            steering_fig = create_steering_trace_plot(steering_trace)
            yield "", final_history, status, wealth_fig, steering_fig
        
        def clear_all():
            """Clear chat and reset plots."""
            empty_wealth_plot = create_wealth_distribution_plot({})
            empty_steering_plot = create_steering_trace_plot({})
            return [], "", empty_wealth_plot, empty_steering_plot
        
        msg_input.submit(
            respond_stream,
            [msg_input, chatbot, temperature, max_tokens, steering_strength, show_stats],
            [msg_input, chatbot, stats_display, wealth_plot, steering_plot]
        )
        
        send_btn.click(
            respond_stream,
            [msg_input, chatbot, temperature, max_tokens, steering_strength, show_stats],
            [msg_input, chatbot, stats_display, wealth_plot, steering_plot]
        )
        
        clear_btn.click(clear_all, outputs=[chatbot, stats_display, wealth_plot, steering_plot])
        
        # Initialize with empty list for messages format
        demo.load(lambda: [], outputs=[chatbot])
        
        swarm_btn.click(get_swarm_status, outputs=swarm_display)
        homeo_btn.click(get_homeostasis_status, outputs=homeo_display)
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
