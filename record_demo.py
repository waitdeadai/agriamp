"""
AgriAMP — Demo Video Recorder
Automates browser with Playwright and records video of the Streamlit app.

Requirements:
    pip install playwright
    playwright install chromium

Usage:
    Terminal 1:  streamlit run app.py
    Terminal 2:  python record_demo.py

Video saved to ./demo_video/
Merge with audio:
    ffmpeg -i demo_video/<file>.webm -i audio.wav -c:v libx264 -crf 18 -c:a aac -b:a 192k -shortest demo_final.mp4
"""

from playwright.sync_api import sync_playwright
import time
import os
import glob

STREAMLIT_URL = "http://localhost:8501"
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_video")
WIDTH, HEIGHT = 1920, 1080

# ─── Timings (seconds) ─── adjust to sync with recorded audio
# Structure based on winning hackathon pitch research:
# Data shock → What+Why now → AMPs 15s → Demo 45s → Proof → Metrics → Market → Close
T = {
    # Scene 1: Data Shock + What/Why Now (0:00 - 0:30)
    "load_wait": 4,             # Wait for Streamlit to render
    "hook_hold": 26,            # Welcome screen with hero stats (narrate data shock + what/why now)

    # Scene 2: AMPs in 15 sec (0:30 - 0:45)
    "amp_expander": 13,         # Open expander, read content, lock/door metaphor
    "amp_close": 2,             # Close expander

    # Scene 3: Live pipeline demo (0:45 - 1:30)
    "pre_click": 2,             # Pause before click
    "pipeline_running": 15,     # Pipeline running (~8-15s GPU)
    "pipeline_result": 8,       # Show "Pipeline completed" + results
    "post_pipeline": 5,         # Workflow log visible, "160 candidates..."

    # Scene 4: Proof — Epinecidin (1:30 - 2:00)
    "caso_real_metrics": 7,     # Crop loss metrics at top
    "caso_real_scroll_1": 2,    # Scroll to validated AMPs table
    "caso_real_table": 8,       # Table with Epinecidin-1 and EPI-4
    "caso_real_scroll_2": 2,    # Scroll to "Epinecidin in our results"
    "caso_real_result": 8,      # EPI-4 > Epinecidin-1

    # Scene 5: Benchmark (2:00 - 2:20)
    "benchmark_table": 8,       # Comparison table
    "benchmark_scroll": 2,      # Scroll to bar chart
    "benchmark_chart": 8,       # AUC bar chart

    # Scene 6: Validation + Market + Close (2:20 - 3:00)
    "validacion_metrics": 7,    # 6 metrics at top (AUC, MCC, Acc, F1, Sens, Spec)
    "validacion_scroll": 2,     # Scroll to charts
    "validacion_charts": 12,    # ROC + confusion matrix (narrate market over this)
    "final_hold": 8,            # Final hold — tagline + silence
}


def smooth_scroll(page, pixels, duration_ms=800):
    """Smooth scroll with quadratic easing."""
    page.evaluate(f"""
        new Promise(resolve => {{
            const start = window.scrollY;
            const target = start + {pixels};
            const duration = {duration_ms};
            const startTime = performance.now();
            function step(currentTime) {{
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const ease = progress < 0.5
                    ? 2 * progress * progress
                    : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                window.scrollTo(0, start + (target - start) * ease);
                if (progress < 1) requestAnimationFrame(step);
                else resolve();
            }}
            requestAnimationFrame(step);
        }})
    """)
    time.sleep(duration_ms / 1000 + 0.2)


def scroll_to_top(page):
    """Scroll back to top of page."""
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    time.sleep(0.5)


def scroll_tabs_into_view(page):
    """Scroll so the tab bar sits near the top of the viewport."""
    page.evaluate("""
        const tabs = document.querySelector('[data-baseweb="tab-list"]');
        if (tabs) {
            tabs.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
    """)
    time.sleep(0.8)


def click_tab(page, tab_text):
    """Click a Streamlit tab by text, ensuring tabs are visible first."""
    scroll_tabs_into_view(page)
    page.locator('button[data-baseweb="tab"]').filter(has_text=tab_text).click()
    time.sleep(1.5)


def click_sidebar_expander(page, text):
    """Toggle sidebar expander."""
    page.locator('[data-testid="stSidebar"] [data-testid="stExpander"]').filter(
        has_text=text
    ).locator('summary, [data-testid="stExpanderToggle"], div[role="button"]').first.click()
    time.sleep(0.5)


def elapsed(start):
    """Format mm:ss from start."""
    s = int(time.time() - start)
    return f"{s // 60}:{s % 60:02d}"


def run_demo():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    total_secs = sum(T.values())
    print(f"AgriAMP Demo Recorder")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print(f"Estimated duration: {total_secs // 60}:{total_secs % 60:02d}")
    print(f"Make sure Streamlit is running at {STREAMLIT_URL}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=1,
        )
        page = context.new_page()
        t0 = time.time()

        # ═══════════════════════════════════════
        # INITIAL LOAD
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Loading app...")
        page.goto(STREAMLIT_URL, wait_until="networkidle")
        page.wait_for_timeout(T["load_wait"] * 1000)

        # ═══════════════════════════════════════
        # SCENE 1: DATA SHOCK + WHAT/WHY NOW (0:00 - 0:30)
        # Screen: hero box with "From your crop to the biological solution"
        # Narration: 50-80% loss, 500M liters, EU 2026
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 1: Data shock — welcome screen")
        time.sleep(T["hook_hold"])

        # ═══════════════════════════════════════
        # SCENE 2: AMPs IN 15 SECONDS (0:30 - 0:45)
        # Open sidebar expander, lock/door metaphor
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 2: AMPs — sidebar expander")
        try:
            click_sidebar_expander(page, "antimicrobial peptides")
        except Exception:
            page.locator('text=What are antimicrobial peptides').first.click()

        time.sleep(T["amp_expander"])

        # Close expander
        try:
            click_sidebar_expander(page, "antimicrobial peptides")
        except Exception:
            page.locator('text=What are antimicrobial peptides').first.click()
        time.sleep(T["amp_close"])

        # ═══════════════════════════════════════
        # SCENE 3: LIVE PIPELINE DEMO (0:45 - 1:30)
        # Click "Run AgriAMP Pipeline" → 6 animated steps
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 3: Live pipeline")
        time.sleep(T["pre_click"])

        pipeline_btn = page.locator('button:has-text("Run AgriAMP Pipeline")')
        precomp_check = page.locator('[data-testid="stSidebar"]').locator(
            'text=Load precomputed results'
        )

        if pipeline_btn.is_visible(timeout=2000):
            pipeline_btn.click()
            print(f"  [{elapsed(t0)}] Pipeline running with GPU...")
        elif precomp_check.is_visible(timeout=2000):
            precomp_check.click()
            print(f"  [{elapsed(t0)}] Fallback: precomputed data")
        else:
            print(f"  [{elapsed(t0)}] WARN: button/checkbox not found")

        # Wait for pipeline
        time.sleep(T["pipeline_running"])

        try:
            page.wait_for_selector('text=Pipeline completed', timeout=45000)
            print(f"  [{elapsed(t0)}] Pipeline completed!")
        except Exception:
            try:
                page.wait_for_selector('text=Results for', timeout=10000)
                print(f"  [{elapsed(t0)}] Results loaded")
            except Exception:
                print(f"  [{elapsed(t0)}] WARN: timeout, continuing...")

        time.sleep(T["pipeline_result"])

        # Scroll down so results are visible (hero + pre-tab content pushes tabs below fold)
        scroll_tabs_into_view(page)
        time.sleep(1)

        # Expand workflow log to show 6 agentic steps with checkmarks
        try:
            workflow_expander = page.locator('[data-testid="stExpander"]').filter(
                has_text="steps completed"
            )
            if workflow_expander.is_visible(timeout=5000):
                workflow_expander.first.click()
                print(f"  [{elapsed(t0)}] Workflow steps expanded")
                time.sleep(1)
                # Scroll down to show all 6 steps
                smooth_scroll(page, 300)
        except Exception:
            print(f"  [{elapsed(t0)}] WARN: could not expand workflow log")

        time.sleep(T["post_pipeline"])

        # ═══════════════════════════════════════
        # SCENE 4: PROOF — EPINECIDIN (1:30 - 2:00)
        # Tab "Real Case": crop loss data + validated AMPs table + Epinecidin in results
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 4: Real Case — Epinecidin")
        click_tab(page, "Real Case")

        # Crop loss metrics at top
        time.sleep(T["caso_real_metrics"])

        # Scroll to validated AMPs table
        smooth_scroll(page, 400)
        time.sleep(T["caso_real_scroll_1"])
        time.sleep(T["caso_real_table"])

        # Scroll to "Epinecidin in our results"
        smooth_scroll(page, 350)
        time.sleep(T["caso_real_scroll_2"])
        time.sleep(T["caso_real_result"])

        # ═══════════════════════════════════════
        # SCENE 5: BENCHMARK vs SOTA (2:00 - 2:20)
        # Tab "Benchmark vs SOTA": comparison table + AUC bar chart
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 5: Benchmark")
        click_tab(page, "Benchmark")

        time.sleep(T["benchmark_table"])

        smooth_scroll(page, 350)
        time.sleep(T["benchmark_scroll"])
        time.sleep(T["benchmark_chart"])

        # ═══════════════════════════════════════
        # SCENE 6: ML VALIDATION + MARKET + CLOSE (2:20 - 3:00)
        # Tab "ML Validation": 6 metrics + ROC + confusion matrix
        # Narration: metrics → market → close
        # ═══════════════════════════════════════
        print(f"[{elapsed(t0)}] Scene 6: ML Validation + Close")
        click_tab(page, "ML Validation")

        # 6 metrics at top
        time.sleep(T["validacion_metrics"])

        # Scroll to ROC + confusion matrix
        smooth_scroll(page, 300)
        time.sleep(T["validacion_scroll"])

        # Charts visible — narrate market and close over this
        time.sleep(T["validacion_charts"])

        # Final hold — tagline + silence
        print(f"[{elapsed(t0)}] Final hold...")
        time.sleep(T["final_hold"])

        # ═══════════════════════════════════════
        # END
        # ═══════════════════════════════════════
        total = time.time() - t0
        print(f"[{elapsed(t0)}] Recording finished ({total:.0f}s)")
        context.close()
        browser.close()

    # Find generated video
    videos = glob.glob(os.path.join(VIDEO_DIR, "*.webm"))
    if videos:
        latest = max(videos, key=os.path.getmtime)
        size_mb = os.path.getsize(latest) / (1024 * 1024)
        print(f"\nVideo: {latest} ({size_mb:.1f} MB)")
        print(f"\nMerge with audio:")
        print(f'  ffmpeg -i "{latest}" -i your_audio.wav \\')
        print(f'    -c:v libx264 -crf 18 -preset slow \\')
        print(f'    -c:a aac -b:a 192k -shortest \\')
        print(f'    demo_agriamp_final.mp4')
    else:
        print(f"\nNo video found in {VIDEO_DIR}/")


if __name__ == "__main__":
    run_demo()
