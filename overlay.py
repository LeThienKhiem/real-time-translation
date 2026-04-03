"""
Overlay Window Module — Streaming Edition
Displays live subtitles as a floating overlay on screen.
Supports partial (in-progress) text + finalized translated lines.
Always on top, semi-transparent, draggable.
"""

import tkinter as tk
import threading
import queue


class SubtitleOverlay:
    """A floating overlay window that displays streaming subtitles."""

    def __init__(self, position="bottom", opacity=0.85, font_size=18, bg_color="#1a1a2e",
                 text_color="#ffffff", partial_color="#00d4ff", translated_color="#00ff88",
                 width=900, height=150):
        self.position = position
        self.opacity = opacity
        self.font_size = font_size
        self.bg_color = bg_color
        self.text_color = text_color
        self.partial_color = partial_color
        self.translated_color = translated_color
        self.width = width
        self.height = height

        self._message_queue = queue.Queue()
        self._running = False
        self._thread = None
        self._root = None
        self._minimized = False

        # Drag state
        self._drag_x = 0
        self._drag_y = 0

    def _create_window(self):
        self._root = tk.Tk()
        self._root.title("Real-time Translator")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self.opacity)
        self._root.configure(bg=self.bg_color)

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        x = (screen_w - self.width) // 2
        if self.position == "top":
            y = 30
        elif self.position == "center":
            y = (screen_h - self.height) // 2
        else:
            y = screen_h - self.height - 60

        self._root.geometry(f"{self.width}x{self.height}+{x}+{y}")

        main_frame = tk.Frame(self._root, bg=self.bg_color, padx=15, pady=6)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = tk.Frame(main_frame, bg=self.bg_color)
        header.pack(fill=tk.X)

        self._status_label = tk.Label(
            header, text="● LIVE", fg="#00ff88", bg=self.bg_color,
            font=("Consolas", 9, "bold"), anchor="w"
        )
        self._status_label.pack(side=tk.LEFT)

        self._lang_label = tk.Label(
            header, text="", fg="#888888", bg=self.bg_color,
            font=("Consolas", 9), anchor="w"
        )
        self._lang_label.pack(side=tk.LEFT, padx=(10, 0))

        close_btn = tk.Label(
            header, text="✕", fg="#ff4444", bg=self.bg_color,
            font=("Arial", 12, "bold"), cursor="hand2"
        )
        close_btn.pack(side=tk.RIGHT)
        close_btn.bind("<Button-1>", lambda e: self.stop())

        min_btn = tk.Label(
            header, text="─", fg="#888888", bg=self.bg_color,
            font=("Arial", 12, "bold"), cursor="hand2"
        )
        min_btn.pack(side=tk.RIGHT, padx=(0, 8))
        min_btn.bind("<Button-1>", lambda e: self._toggle_minimize())

        # Content area
        self._content_frame = tk.Frame(main_frame, bg=self.bg_color)
        self._content_frame.pack(fill=tk.BOTH, expand=True)

        # Line 1: Live partial transcription (what's being heard right now)
        self._partial_label = tk.Label(
            self._content_frame, text="Đang chờ giọng nói...",
            fg=self.partial_color, bg=self.bg_color,
            font=("Segoe UI", self.font_size - 2, "italic"),
            wraplength=self.width - 40, anchor="w", justify="left"
        )
        self._partial_label.pack(fill=tk.X, pady=(4, 0))

        # Line 2: Latest finalized translation
        self._translated_label = tk.Label(
            self._content_frame, text="",
            fg=self.translated_color, bg=self.bg_color,
            font=("Segoe UI", self.font_size, "bold"),
            wraplength=self.width - 40, anchor="w", justify="left"
        )
        self._translated_label.pack(fill=tk.X, pady=(2, 0))

        # Line 3: Previous translation (dimmed)
        self._prev_label = tk.Label(
            self._content_frame, text="",
            fg="#666666", bg=self.bg_color,
            font=("Segoe UI", self.font_size - 4),
            wraplength=self.width - 40, anchor="w", justify="left"
        )
        self._prev_label.pack(fill=tk.X, pady=(0, 0))

        # Make window draggable
        for widget in [main_frame, header, self._status_label, self._lang_label]:
            widget.bind("<Button-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

        self._process_messages()

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        x = self._root.winfo_x() + event.x - self._drag_x
        y = self._root.winfo_y() + event.y - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _toggle_minimize(self):
        if not self._minimized:
            self._content_frame.pack_forget()
            self._root.geometry(f"{self.width}x{30}")
            self._minimized = True
        else:
            self._content_frame.pack(fill=tk.BOTH, expand=True)
            self._root.geometry(f"{self.width}x{self.height}")
            self._minimized = False

    def _process_messages(self):
        try:
            while not self._message_queue.empty():
                msg = self._message_queue.get_nowait()
                msg_type = msg.get("type")

                if msg_type == "partial":
                    # Live partial text (updates frequently)
                    self._partial_label.config(text=msg.get("text", ""))

                elif msg_type == "finalized":
                    # A sentence is finalized: show original + translation
                    original = msg.get("original", "")
                    translated = msg.get("translated", "")

                    # Move current translated to prev
                    current = self._translated_label.cget("text")
                    if current:
                        self._prev_label.config(text=current)

                    self._partial_label.config(text=original, fg=self.text_color,
                                                font=("Segoe UI", self.font_size - 3))
                    self._translated_label.config(text=translated)

                elif msg_type == "clear_partial":
                    self._partial_label.config(
                        text="", fg=self.partial_color,
                        font=("Segoe UI", self.font_size - 2, "italic")
                    )

                elif msg_type == "status":
                    self._status_label.config(
                        text=f"● {msg.get('status', '')}", fg=msg.get("color", "#00ff88")
                    )

                elif msg_type == "lang":
                    self._lang_label.config(text=msg.get("text", ""))

                elif msg_type == "stop":
                    self._running = False
                    if self._root:
                        self._root.destroy()
                    return
        except Exception:
            pass

        if self._running and self._root:
            self._root.after(30, self._process_messages)  # 30ms = ~33fps update rate

    # --- Public API (all thread-safe via queue) ---

    def show_partial(self, text):
        """Show live partial transcription (updates rapidly)."""
        self._message_queue.put({"type": "partial", "text": text})

    def show_finalized(self, original, translated):
        """Show finalized original + translated text."""
        self._message_queue.put({"type": "finalized", "original": original, "translated": translated})

    def clear_partial(self):
        """Clear the partial text line."""
        self._message_queue.put({"type": "clear_partial"})

    def update_status(self, status, color="#00ff88"):
        self._message_queue.put({"type": "status", "status": status, "color": color})

    def update_lang(self, text):
        self._message_queue.put({"type": "lang", "text": text})

    # Keep old API for compatibility
    def update_subtitle(self, original="", translated="", source_lang="", target_lang=""):
        self.show_finalized(original, translated)
        if source_lang and target_lang:
            self.update_lang(f"{source_lang} → {target_lang}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._create_window()
        if self._root:
            self._root.mainloop()

    def stop(self):
        self._message_queue.put({"type": "stop"})
        self._running = False

    def is_running(self):
        return self._running
