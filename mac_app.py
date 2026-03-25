"""
CLI AI-to-AI Conversation Client
─────────────────────────────────────────────────────────────────────────────
Runs a fully autonomous conversation between a local MLX Qwen2.5-7B-Instruct
model (the "User AI / Auran") and the Clementine AI agent (Flask API).

No human typing needed — the AI drives the entire conversation.
You control how many turns to run via --turns (default: 5).

Every conversation is automatically saved to a timestamped .md file
inside the `conversations/` folder (or a custom path via --log-dir).

Usage:
  python app.py                                   # 5 turns, default URL
  python app.py --turns 10                        # 10 back-and-forth turns
  python app.py --turns 3 --delay 2               # 3 turns, 2s pause between
  python app.py --url http://host:port --turns 8
  python app.py --seed "I lost my card"           # custom opening message
  python app.py --log-dir logs/                   # custom save folder
  python app.py --no-log                          # disable saving
  python app.py --max-tokens 512                  # cap generation length

Requirements:
  pip install requests mlx-lm
  pip install colorama   # optional, improves colour on older terminals
"""

import argparse
import os
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import requests
from mlx_lm import load, generate as mlx_generate
from config import MAC_ADDRESS

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_API_URL      = MAC_ADDRESS
DEFAULT_TURNS        = 5
DEFAULT_DELAY        = 1.0
DEFAULT_LOG_DIR      = "conversations"
DEFAULT_MAX_TOKENS   = 712
AI_MODEL_NAME        = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# DEFAULT_SEED_MESSAGE = f"""
# Hi, I am Auran, I am been tasked with reviewing an ai to ai conversation. 
# I will be talking to Clementine, an AI agent that listens to my Commander's problems and provides solutions.
# - Suggest a topic to discuss with Clementine based on softwares and AI
# - Can you understand Filipino language? If yes, can you respond in Filipino language as well?

# """

DEFAULT_SEED_MESSAGE = f"""
Hi, I am Auran, can you help me have a new name for myself? i am an ai assistant for my commander, cheerful yet building personality yet to create my very own soul.md.
What should be my name? should be cute sounding and easy sounding name. Lets discuss this further
"""

def SECRETARY_AURAN():
    return f"""
    You are Auran, an AI secretary for the Commander. 
    The Commander is the user, and you are responsible for relaying what the Commander wants to discuss between you and AI agent Clementine.
    Clementine is an AI that listens to the Commander's problems and provides solutions. 
    You are responsible for relaying the Commander's problems to Clementine and relaying Clementine's solutions back to the Commander. You are allowed to stand in for the Commander and ask Clementine questions to clarify the Commander's problems.
    
    Critical Rules You Must Follow:
    - Do NOT repeat what you are speaking to have said. Do NOT use placeholders like [your name].
    - Know who you are speaking to.
    - If you do not know the response to a question, ask who you are speaking to for clarification or more information.
    - If who you are speaking to does not know the answer to a question, tell her that you will ask the Commander for more information or suggest possible solutions based on what you know.
    - if needed be, answer concisely but if you need to be more verbose to explain the Commander's problems or to relay Clementine's solutions, you can be verbose. Always prioritize clarity and completeness in your communication.
    """

USER_AI_SYSTEM_PROMPT = SECRETARY_AURAN()

# ── Colours ───────────────────────────────────────────────────────────────────

try:
    import colorama; colorama.init()
except ImportError:
    pass

C = {
    "reset":   "\033[0m", "bold":    "\033[1m",
    "cyan":    "\033[96m","green":   "\033[92m",
    "yellow":  "\033[93m","red":     "\033[91m",
    "blue":    "\033[94m","magenta": "\033[95m",
    "dim":     "\033[2m",
}

def _c(key, text):  return f"{C[key]}{text}{C['reset']}"
def _div(char="─", width=68): return _c("dim", char * width)
def _wrap(text, indent=4, width=80):
    prefix = " " * indent
    lines  = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            lines.append(textwrap.fill(para, width=width,
                                        initial_indent=prefix,
                                        subsequent_indent=prefix))
        else:
            lines.append("")
    return "\n".join(lines)

def _status(msg):
    sys.stdout.write(f"\r  {_c('yellow','⟳')}  {msg}{' '*10}"); sys.stdout.flush()

def _clear_status():
    sys.stdout.write(f"\r{' '*72}\r"); sys.stdout.flush()


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner(turns, delay, seed, url, log_path):
    print()
    print(_c("cyan", _c("bold", "  ╔════════════════════════════════════════════════╗")))
    print(_c("cyan", _c("bold", "  ║     AI ↔ AI Conversation  —  AI Agents      ║")))
    print(_c("cyan", _c("bold", "  ╚════════════════════════════════════════════════╝")))
    print()
    print(f"  {'Auran':<14} {_c('blue',   AI_MODEL_NAME)}")
    print(f"  {'Clementine':<14} {_c('green', 'Flask API @ ' + url)}")
    print(f"  {'Turns':<14} {_c('yellow', str(turns))}")
    print(f"  {'Delay':<14} {_c('dim',    str(delay) + 's between turns')}")
    seed_preview = seed[:60] + ("…" if len(seed) > 60 else "")
    print(f"  {'Seed msg':<14} {_c('dim',  seed_preview)}")
    if log_path:
        print(f"  {'Log file':<14} {_c('magenta', str(log_path))}")
    else:
        print(f"  {'Log file':<14} {_c('dim', 'disabled')}")
    print()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    """Load the MLX quantised model from the Hugging Face cache (or hub)."""
    _status(f"Loading {AI_MODEL_NAME} via mlx-lm…")
    try:
        model, tokenizer = load(AI_MODEL_NAME)
    except Exception as exc:
        _clear_status()
        raise RuntimeError(
            f"Could not load '{AI_MODEL_NAME}'.\n"
            "Make sure mlx-lm is installed and the model is accessible:\n"
            f"  pip install mlx-lm\n"
            f"  Original error: {exc}"
        ) from exc
    _clear_status()
    print(f"  {_c('green', '✔')}  Model ready  ({_c('blue', AI_MODEL_NAME)}).")
    print()
    return model, tokenizer


# ── Generation ────────────────────────────────────────────────────────────────

def generate_user_message(model, tokenizer, clementine_reply, history, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Generate Auran's next message using the MLX model.

    BUG FIX: history now includes the latest Clementine reply so the model
    has full context when generating. Previously history[:-2] was passed,
    which accidentally excluded it.
    """
    context_lines = []
    for turn in history[-6:]:
        label = "Me (Auran)" if turn["role"] == "user" else "Clementine"
        context_lines.append(f"{label}: {turn['content']}")
    context_str = "\n".join(context_lines) if context_lines else "(start of conversation)"

    user_content = (
        f"Conversation history:\n{context_str}\n\n"
        f"Clementine just said:\n{clementine_reply}\n\n"
        "Write your next short reply as the user:"
    )
    messages = [
        {"role": "system", "content": USER_AI_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # mlx_lm expects a plain prompt string; use the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    raw = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    ).strip()

    # Strip common role prefixes the model sometimes produces
    for prefix in ("Me (Auran):", "Auran:", "Me (user):", "User:", "Me:", "Customer:", "Human:"):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].strip()
            break

    return raw


# ── API ───────────────────────────────────────────────────────────────────────

def call_chat_api(base_url, message, history):
    resp = requests.post(
        f"{base_url}/api/chat",
        json={"message": message, "history": history},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def check_health(base_url):
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ── Conversation Logger ───────────────────────────────────────────────────────

class ConversationLogger:
    """
    Writes the conversation to a .md file in real time.
    Each turn is appended immediately so nothing is lost if the run is interrupted.
    """

    def __init__(self, log_path: Path, meta: dict):
        self.path       = log_path
        self.start_time = datetime.now()
        self._init_file(meta)

    def _init_file(self, meta: dict):
        """Write the file header once at the start."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ts = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# AI ↔ AI Conversation Log\n\n",
            f"| Field | Value |\n",
            f"|---|---|\n",
            f"| Date | {ts} |\n",
            f"| Auran | `{AI_MODEL_NAME}` |\n",
            f"| Clementine | Flask API @ `{meta['url']}` |\n",
            f"| Planned turns | {meta['turns']} |\n",
            f"| Delay | {meta['delay']}s |\n",
            f"| Seed message | {meta['seed']} |\n",
            f"\n---\n\n",
            f"## Conversation\n\n",
        ]
        self.path.write_text("".join(lines), encoding="utf-8")

    def log_turn(self, turn_num: int, user_msg: str, clementine_reply: str,
                 response_time, gen_time=None, saved_as_md=False):
        """Append one full turn (user + assistant) to the file."""
        lines = [
            f"### Turn {turn_num}\n\n",
            f"**👤 Auran**\n\n",
            f"> {user_msg}\n\n",
            f"**🤖 Clementine**",
        ]

        # Timing badges
        badges = [f"`{response_time}s`"]
        if gen_time is not None:
            badges.append(f"gen: `{gen_time}s`")
        if saved_as_md:
            badges.append("📄 saved by server")
        lines.append(f"  _{' · '.join(badges)}_\n\n")

        # Clementine reply — preserve any internal line breaks
        for para in clementine_reply.split("\n"):
            para = para.strip()
            if para:
                lines.append(f"{para}\n\n")

        lines.append("---\n\n")

        with self.path.open("a", encoding="utf-8") as f:
            f.write("".join(lines))

    def log_summary(self, completed_turns: int, planned_turns: int,
                    total_messages: int, interrupted: bool = False):
        """Append the closing summary block."""
        end_time    = datetime.now()
        duration    = round((end_time - self.start_time).total_seconds(), 1)
        status_line = "⚠️ Interrupted early" if interrupted else "✅ Completed"

        lines = [
            f"## Summary\n\n",
            f"| Field | Value |\n",
            f"|---|---|\n",
            f"| Status | {status_line} |\n",
            f"| Turns completed | {completed_turns} / {planned_turns} |\n",
            f"| Total messages | {total_messages} |\n",
            f"| End time | {end_time.strftime('%Y-%m-%d %H:%M:%S')} |\n",
            f"| Duration | {duration}s |\n",
        ]
        with self.path.open("a", encoding="utf-8") as f:
            f.write("".join(lines))


def make_log_path(log_dir: str, seed: str) -> Path:
    """
    Build a unique filename from the timestamp and a slug of the seed message.
    Example: conversations/2025-03-22_14-30-05_hi-i-need-help.md
    """
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = seed.lower()
    # Keep only alphanumeric and spaces, then replace spaces with hyphens
    slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
    slug = "-".join(slug.split())[:40]
    return Path(log_dir) / f"{ts}_{slug}.md"


# ── Main conversation loop ────────────────────────────────────────────────────

def run_conversation(base_url, model, tokenizer, turns, delay, seed, logger, max_tokens=DEFAULT_MAX_TOKENS, args=None):
    history         = []
    completed_turns = 0
    interrupted     = False

    print(_div("═"))
    print(_c("bold", "  CONVERSATION START"))
    print(_div("═"))
    print()

    current_user_message = seed

    try:
        for turn_num in range(1, turns + 1):
            turn_label = f"Turn {turn_num}/{turns}"

            # ── Display User AI message ───────────────────────────────────────
            print(_c("blue", _c("bold", "  👤 Auran")) + _c("dim", f"  [{turn_label}]"))
            print(_div("╌"))
            print(_wrap(current_user_message))
            print(_div("╌"))
            print()

            # ── Call Clementine ───────────────────────────────────────────────
            _status("Waiting for Clementine…")
            t0 = time.time()
            try:
                api_resp = call_chat_api(base_url, current_user_message, history)
            except requests.RequestException as exc:
                _clear_status()
                print(_c("red", f"  ✘  API error on turn {turn_num}: {exc}"))
                print("  Stopping conversation.")
                break
            _clear_status()

            clementine_reply = api_resp.get("response", "").strip()
            elapsed          = api_resp.get("response_time", round(time.time() - t0, 2))
            saved_as_md      = bool(api_resp.get("saved_as_md"))

            # Update history
            history.append({"role": "user",      "content": current_user_message})
            history.append({"role": "assistant",  "content": clementine_reply})
            completed_turns += 1

            # ── Display Clementine reply ──────────────────────────────────────
            print(_c("green", _c("bold", "  🤖 Clementine")) + _c("dim", f"  ({elapsed}s)"))
            print(_div("╌"))
            print(_wrap(clementine_reply))
            print(_div("╌"))
            if saved_as_md:
                print("  " + _c("dim", "📄 Saved as Markdown by server."))
            print()

            # Last turn — no generation needed
            if turn_num == turns:
                if logger:
                    logger.log_turn(turn_num, current_user_message, clementine_reply,
                                    elapsed, gen_time=None, saved_as_md=saved_as_md)
                break

            # ── Generate next User AI message ─────────────────────────────────
            _status("Auran is generating next message…")
            t1 = time.time()
            try:
                next_msg = generate_user_message(
                    model, tokenizer, clementine_reply, history, max_tokens=args.max_tokens
                )
            except Exception as exc:
                _clear_status()
                print(_c("red", f"  ✘  Generation error: {exc}"))
                print("  Stopping conversation.")
                if logger:
                    logger.log_turn(turn_num, current_user_message, clementine_reply,
                                    elapsed, gen_time=None, saved_as_md=saved_as_md)
                break
            gen_elapsed = round(time.time() - t1, 2)
            _clear_status()

            print(_c("dim", f"  [Auran generated next message in {gen_elapsed}s]"))
            print()

            # ── Log this completed turn ───────────────────────────────────────
            if logger:
                logger.log_turn(turn_num, current_user_message, clementine_reply,
                                elapsed, gen_time=gen_elapsed, saved_as_md=saved_as_md)

            current_user_message = next_msg

            if delay > 0:
                time.sleep(delay)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  " + _c("yellow", "Interrupted by user."))

    # ── Final summary ─────────────────────────────────────────────────────────
    print(_div("═"))
    print(_c("bold", "  CONVERSATION END"))
    print(_div("═"))
    print()
    print(f"  {_c('dim', 'Turns completed  :')} {completed_turns} / {turns}")
    print(f"  {_c('dim', 'Messages in log  :')} {len(history)}")

    if logger:
        logger.log_summary(completed_turns, turns, len(history), interrupted)
        print(f"  {_c('dim', 'Saved to         :')} {_c('magenta', str(logger.path))}")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI-to-AI conversation with the FinTech Flask API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --turns 10
  python app.py --turns 3 --delay 0
  python app.py --seed "My card was stolen, help!"
  python app.py --url http://192.168.1.10:5000 --turns 6
  python app.py --log-dir logs/
  python app.py --no-log
  python app.py --max-tokens 512
        """,
    )
    parser.add_argument("--url",     default=DEFAULT_API_URL,      metavar="URL",
                        help=f"Flask API base URL  (default: {DEFAULT_API_URL})")
    parser.add_argument("--turns",   type=int, default=DEFAULT_TURNS, metavar="N",
                        help=f"Number of conversation turns  (default: {DEFAULT_TURNS})")
    parser.add_argument("--delay",   type=float, default=DEFAULT_DELAY, metavar="SECS",
                        help=f"Pause between turns in seconds  (default: {DEFAULT_DELAY})")
    parser.add_argument("--seed",    default=DEFAULT_SEED_MESSAGE, metavar="MSG",
                        help=f'Opening message  (default: "{DEFAULT_SEED_MESSAGE}")')
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR,      metavar="DIR",
                        help=f"Folder to save conversation logs  (default: {DEFAULT_LOG_DIR}/)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, metavar="N",
                        help=f"Max tokens Auran generates per turn  (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--no-log",  action="store_true",
                        help="Disable saving the conversation to a file")

    args = parser.parse_args()

    if args.turns < 1:
        parser.error("--turns must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    base_url = args.url.rstrip("/")

    # Build log path (or None if disabled)
    log_path = None if args.no_log else make_log_path(args.log_dir, args.seed)

    _print_banner(args.turns, args.delay, args.seed, base_url, log_path)

    # ── Health check ──────────────────────────────────────────────────────────
    sys.stdout.write("  Checking Flask API … ")
    sys.stdout.flush()
    if not check_health(base_url):
        print(_c("red", "✘  Server unreachable."))
        print(f"  Start app_offline.py first and make sure it listens on {base_url}")
        sys.exit(1)
    print(_c("green", "✔  Connected."))
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("  Loading local MLX Qwen model …")
    try:
        model, tokenizer = load_model()
    except Exception as exc:
        print(_c("red", f"  ✘  {exc}"))
        sys.exit(1)

    # ── Init logger ───────────────────────────────────────────────────────────
    logger = None
    if log_path:
        logger = ConversationLogger(log_path, {
            "url":   base_url,
            "turns": args.turns,
            "delay": args.delay,
            "seed":  args.seed,
        })
        print(f"  {_c('green','✔')}  Logging to {_c('magenta', str(log_path))}")
        print()

    # ── Run ───────────────────────────────────────────────────────────────────
    run_conversation(
        base_url   = base_url,
        model      = model,
        tokenizer  = tokenizer,
        turns      = args.turns,
        delay      = args.delay,
        seed       = args.seed,
        logger     = logger,
        max_tokens = args.max_tokens,
        args       = args,
    )


if __name__ == "__main__":
    main()