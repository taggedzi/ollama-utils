import queue
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .common import detect_ollama_version
from .test_models import DEFAULT_REPORT_PATH
from .test_models import OLLAMA_API_BASE_URL
from .test_models import normalize_ollama_api_base_url
from .test_models import main as test_main
from .update_models import main as update_main

APP_BG = "#F4EFE6"
PANEL_BG = "#FBF7F1"
ACCENT = "#A44A3F"
TEXT_DARK = "#1E1B18"
TEXT_MUTED = "#5B5147"
LOG_BG = "#16181C"
LOG_FG = "#F5EBDD"
REPO_URL = "https://github.com/taggedzi/tz-ollama-utils"


def _asset_roots():
    roots = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        roots.append(Path(bundle_root) / "assets")
        roots.append(Path(bundle_root) / "tz_ollama_utils" / "assets")

    repo_assets = Path(__file__).resolve().parents[2] / "assets"
    package_assets = Path(__file__).resolve().parent / "assets"
    roots.extend([repo_assets, package_assets])

    deduped_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped_roots.append(root)
            seen.add(key)
    return deduped_roots


def _find_asset(*parts):
    for root in _asset_roots():
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def _configure_windows_app_id():
    if not sys.platform.startswith("win"):
        return

    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "taggedz.tz-ollama-utils"
        )
    except (AttributeError, OSError):
        pass


class OllamaUtilsApp:
    def __init__(self, root, tk_module, ttk_module):
        self.tk = tk_module
        self.ttk = ttk_module
        import tkinter.filedialog as _fd
        self.filedialog = _fd
        self.root = root
        self.root.title("Taggedz's Ollama Utilities")
        self.root.geometry("1040x780")
        self.root.minsize(900, 640)
        self.root.configure(bg=APP_BG)

        self.log_queue = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()

        self.timeout_var = self.tk.StringVar(value="300")
        self.ignore_size_var = self.tk.BooleanVar(value=False)
        self.vram_override_var = self.tk.StringVar(value="")
        self.api_base_url_var = self.tk.StringVar(value=OLLAMA_API_BASE_URL)
        self.status_var = self.tk.StringVar(value="Idle")
        self.report_var = self.tk.StringVar(value=f"Report: {DEFAULT_REPORT_PATH.resolve()}")
        self.report_path_var = self.tk.StringVar(value=str(DEFAULT_REPORT_PATH.resolve()))
        self.active_job_var = self.tk.StringVar(value="No job running")
        self.version_var = self.tk.StringVar(
            value=f"App v{__version__} | Ollama version: detecting..."
        )
        self.about_window = None
        self.window_icon = None
        self.header_logo = None

        self._configure_styles()
        self._apply_window_icon()
        self.header_logo = self._load_header_logo()
        self._build_ui()
        self._load_versions()
        self.root.after(100, self._drain_log_queue)

    def _configure_styles(self):
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass

        style.configure("App.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=PANEL_BG, relief="flat")
        style.configure("Header.TFrame", background=ACCENT)
        style.configure(
            "Title.TLabel",
            background=ACCENT,
            foreground="#FFF7EF",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=ACCENT,
            foreground="#F6DDD6",
            font=("Segoe UI", 10),
        )
        style.configure("HeaderLogo.TLabel", background=ACCENT)
        style.configure(
            "SectionTitle.TLabel",
            background=PANEL_BG,
            foreground=TEXT_DARK,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=PANEL_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Meta.TLabel",
            background=APP_BG,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 8),
        )
        style.configure("Quiet.TButton", padding=(10, 7))
        style.configure("App.TNotebook", background=APP_BG, borderwidth=0)
        style.configure(
            "App.TNotebook.Tab",
            background="#D8D0C4",
            foreground=TEXT_MUTED,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "App.TNotebook.Tab",
            background=[
                ("selected", PANEL_BG),
                ("active", "#E7DED2"),
            ],
            foreground=[
                ("selected", TEXT_DARK),
                ("active", TEXT_DARK),
            ],
            padding=[
                ("selected", (24, 14)),
                ("active", (20, 11)),
            ],
        )

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        self._build_header()
        self._build_tabs()
        self._build_activity_area()
        self._build_footer()

    def _build_header(self):
        header = self.ttk.Frame(self.root, style="Header.TFrame", padding=(22, 18))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        if self.header_logo is not None:
            self.ttk.Label(
                header,
                image=self.header_logo,
                style="HeaderLogo.TLabel",
            ).grid(row=0, column=0, rowspan=2, padx=(0, 16), sticky="w")

        self.ttk.Label(
            header,
            text="TaggedZ's Ollama Utilities",
            style="Title.TLabel",
        ).grid(row=0, column=1, sticky="w")
        self.ttk.Label(
            header,
            text=(
                "Update local models, run inventory checks, and capture reports "
                "from a cleaner desktop workflow."
            ),
            style="Subtitle.TLabel",
        ).grid(row=1, column=1, pady=(6, 0), sticky="w")

    def _build_tabs(self):
        shell = self.ttk.Frame(self.root, style="App.TFrame", padding=(18, 16, 18, 10))
        shell.grid(row=1, column=0, sticky="ew")
        shell.columnconfigure(0, weight=1)

        notebook = self.ttk.Notebook(shell, style="App.TNotebook")
        notebook.grid(row=0, column=0, sticky="ew")

        update_tab = self.ttk.Frame(notebook, style="Card.TFrame", padding=18)
        test_tab = self.ttk.Frame(notebook, style="Card.TFrame", padding=18)

        notebook.add(update_tab, text="Update")
        notebook.add(test_tab, text="Test & Report")

        self._build_update_tab(update_tab)
        self._build_test_tab(test_tab)

    def _build_update_tab(self, parent):
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)

        summary = self._make_card(parent)
        summary.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        summary.columnconfigure(0, weight=1)

        self.ttk.Label(summary, text="Update Installed Models", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(
            summary,
            text=(
                "Runs `ollama pull` for every locally installed model and streams live "
                "output into the activity console below."
            ),
            style="Body.TLabel",
            wraplength=480,
            justify="left",
        ).grid(row=1, column=0, pady=(8, 14), sticky="w")

        highlights = [
            "Uses the same update flow as the CLI entry point.",
            "Stop requests finish the current termination step cleanly.",
            "Best used before running a validation pass from the Test tab.",
        ]
        for index, text in enumerate(highlights, start=2):
            self.ttk.Label(
                summary,
                text=f"• {text}",
                style="Body.TLabel",
                wraplength=480,
                justify="left",
            ).grid(row=index, column=0, pady=(0, 8), sticky="w")

        actions = self._make_card(parent)
        actions.grid(row=0, column=1, sticky="nsew")
        actions.columnconfigure(0, weight=1)

        self.ttk.Label(actions, text="Update Actions", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(
            actions,
            text="Launch a full refresh or stop an in-progress pull.",
            style="Body.TLabel",
            wraplength=280,
            justify="left",
        ).grid(row=1, column=0, pady=(8, 16), sticky="w")

        self.update_button = self.ttk.Button(
            actions,
            text="Start Model Update",
            command=self.run_update,
            style="Action.TButton",
        )
        self.update_button.grid(row=2, column=0, pady=(0, 10), sticky="ew")

        self.stop_button = self.ttk.Button(
            actions,
            text="Stop Current Job",
            command=self.stop_run,
            state="disabled",
            style="Quiet.TButton",
        )
        self.stop_button.grid(row=3, column=0, sticky="ew")

    def _build_test_tab(self, parent):
        parent.columnconfigure(0, weight=2)
        parent.columnconfigure(1, weight=3)

        settings = self._make_card(parent)
        settings.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        settings.columnconfigure(1, weight=1)

        self.ttk.Label(settings, text="Test Settings", style="SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.ttk.Label(
            settings,
            text="Control timeout, VRAM policy, and report destination for the inventory run.",
            style="Body.TLabel",
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, pady=(8, 16), sticky="w")

        self.ttk.Label(settings, text="Per-model timeout (seconds)", style="Body.TLabel").grid(
            row=2, column=0, padx=(0, 10), pady=(0, 10), sticky="w"
        )
        self.ttk.Entry(settings, textvariable=self.timeout_var, width=12).grid(
            row=2, column=1, pady=(0, 10), sticky="ew"
        )

        self.ttk.Label(settings, text="Manual VRAM override (MiB)", style="Body.TLabel").grid(
            row=3, column=0, padx=(0, 10), pady=(0, 10), sticky="w"
        )
        self.ttk.Entry(settings, textvariable=self.vram_override_var, width=12).grid(
            row=3, column=1, pady=(0, 10), sticky="ew"
        )

        self.ttk.Label(settings, text="Ollama API base URL", style="Body.TLabel").grid(
            row=4, column=0, padx=(0, 10), pady=(0, 10), sticky="w"
        )
        self.ttk.Entry(settings, textvariable=self.api_base_url_var).grid(
            row=4, column=1, pady=(0, 10), sticky="ew"
        )

        self.ttk.Checkbutton(
            settings,
            text="Ignore VRAM size filter entirely",
            variable=self.ignore_size_var,
        ).grid(row=5, column=0, columnspan=2, pady=(2, 0), sticky="w")

        report = self._make_card(parent)
        report.grid(row=0, column=1, sticky="nsew")
        report.columnconfigure(0, weight=1)
        report.columnconfigure(1, weight=0)

        self.ttk.Label(report, text="Report Output", style="SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.ttk.Label(
            report,
            text=(
                "Choose where the YAML report should be written. Partial reports are "
                "still saved when a test run is stopped."
            ),
            style="Body.TLabel",
            wraplength=460,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, pady=(8, 16), sticky="w")

        self.ttk.Entry(report, textvariable=self.report_path_var).grid(
            row=2, column=0, padx=(0, 8), sticky="ew"
        )
        self.ttk.Button(
            report,
            text="Browse",
            command=self.choose_report_path,
            style="Quiet.TButton",
        ).grid(row=2, column=1, sticky="ew")

        self.test_button = self.ttk.Button(
            report,
            text="Start Test and Inventory",
            command=self.run_test,
            style="Action.TButton",
        )
        self.test_button.grid(row=3, column=0, columnspan=2, pady=(16, 10), sticky="ew")

        self.test_stop_button = self.ttk.Button(
            report,
            text="Stop Test Job",
            command=self.stop_run,
            state="disabled",
            style="Quiet.TButton",
        )
        self.test_stop_button.grid(row=4, column=0, columnspan=2, sticky="ew")

    def _build_activity_area(self):
        activity = self.ttk.Frame(self.root, style="App.TFrame", padding=(18, 0, 18, 10))
        activity.grid(row=2, column=0, sticky="nsew")
        activity.columnconfigure(0, weight=1)
        activity.rowconfigure(1, weight=1)

        status_card = self._make_card(activity, padding=(16, 14))
        status_card.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        status_card.columnconfigure(0, weight=1)
        status_card.columnconfigure(1, weight=0)

        self.ttk.Label(status_card, text="Activity", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Label(status_card, textvariable=self.status_var, style="SectionTitle.TLabel").grid(
            row=0, column=1, sticky="e"
        )
        self.ttk.Label(status_card, textvariable=self.active_job_var, style="Body.TLabel").grid(
            row=1, column=0, pady=(6, 0), sticky="w"
        )
        self.clear_button = self.ttk.Button(
            status_card,
            text="Clear Log",
            command=self.clear_log,
            style="Quiet.TButton",
        )
        self.clear_button.grid(row=1, column=1, pady=(6, 0), sticky="e")

        log_card = self._make_card(activity, padding=(0, 0))
        log_card.grid(row=1, column=0, sticky="nsew")
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(0, weight=1)

        self.log_text = self.tk.Text(
            log_card,
            wrap="word",
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground=LOG_FG,
            relief="flat",
            bd=0,
            padx=14,
            pady=14,
            font=("Consolas", 10),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = self.ttk.Scrollbar(log_card, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _build_footer(self):
        footer = self.ttk.Frame(self.root, style="App.TFrame", padding=(18, 0, 18, 18))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=0)

        self.ttk.Label(footer, textvariable=self.version_var, style="Meta.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.ttk.Button(
            footer,
            text="About",
            command=self.open_about_modal,
            style="Quiet.TButton",
        ).grid(row=0, column=1, sticky="e")
        self.ttk.Label(footer, textvariable=self.report_var, style="Meta.TLabel").grid(
            row=1, column=0, pady=(4, 0), sticky="w"
        )
        self.ttk.Label(
            footer,
            text="Requires local Ollama CLI/API. Generated binaries do not bundle Ollama itself.",
            style="Meta.TLabel",
        ).grid(row=2, column=0, pady=(4, 0), sticky="w")

    def _make_card(self, parent, padding=18):
        return self.ttk.Frame(parent, style="Card.TFrame", padding=padding)

    def _load_header_logo(self):
        logo_path = _find_asset("icons", "tz-ollama-utils-header-88px.png")
        if logo_path is None:
            return None

        try:
            image = self.tk.PhotoImage(file=str(logo_path))
        except self.tk.TclError:
            return None
        return image

    def _apply_window_icon(self):
        bitmap_path = _find_asset("icons", "tz_ollama_utils_icon.ico")
        if bitmap_path is not None and sys.platform.startswith("win"):
            try:
                self.root.iconbitmap(str(bitmap_path))
            except self.tk.TclError:
                pass

        icon_path = _find_asset("icons", "tz-ollama-utils.png")
        if icon_path is not None:
            try:
                self.window_icon = self.tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self.window_icon)
            except self.tk.TclError:
                self.window_icon = None

    def open_about_modal(self):
        if self.about_window is not None and self.about_window.winfo_exists():
            self.about_window.lift()
            self.about_window.focus_force()
            return

        window = self.tk.Toplevel(self.root)
        window.title("About TaggedZ's Ollama Utilities")
        window.transient(self.root)
        window.grab_set()
        window.resizable(False, False)
        window.configure(bg=APP_BG)
        window.protocol("WM_DELETE_WINDOW", self.close_about_modal)
        self.about_window = window

        if self.window_icon is not None:
            try:
                window.iconphoto(True, self.window_icon)
            except self.tk.TclError:
                pass

        container = self.ttk.Frame(window, style="Card.TFrame", padding=20)
        container.grid(row=0, column=0, padx=18, pady=18, sticky="nsew")
        container.columnconfigure(0, weight=1)

        self.ttk.Label(
            container,
            text="TaggedZ's Ollama Utilities",
            style="SectionTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.ttk.Label(
            container,
            text=f"Version {__version__}",
            style="Body.TLabel",
        ).grid(row=1, column=0, pady=(6, 14), sticky="w")
        self.ttk.Label(
            container,
            text=(
                "Desktop tools for updating installed Ollama models, running inventory "
                "checks, and saving YAML reports from one interface."
            ),
            style="Body.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=2, column=0, pady=(0, 10), sticky="w")
        self.ttk.Label(
            container,
            text=(
                "Requires a local or reachable Ollama CLI/API. "
                "This utility does not bundle Ollama itself."
            ),
            style="Body.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=3, column=0, pady=(0, 10), sticky="w")
        repo_link = self.ttk.Label(
            container,
            text=REPO_URL,
            style="Body.TLabel",
            cursor="hand2",
        )
        repo_link.grid(row=4, column=0, pady=(0, 16), sticky="w")
        repo_link.bind("<Button-1>", lambda _event: self.open_repo_link())
        self.ttk.Button(
            container,
            text="Close",
            command=self.close_about_modal,
            style="Quiet.TButton",
        ).grid(row=5, column=0, sticky="e")

        window.update_idletasks()
        parent_x = self.root.winfo_rootx()
        parent_y = self.root.winfo_rooty()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        width = window.winfo_width()
        height = window.winfo_height()
        x = parent_x + max((parent_width - width) // 2, 0)
        y = parent_y + max((parent_height - height) // 2, 0)
        window.geometry(f"+{x}+{y}")
        window.focus_force()

    def close_about_modal(self):
        if self.about_window is None:
            return
        if self.about_window.winfo_exists():
            self.about_window.destroy()
        self.about_window = None

    def open_repo_link(self):
        webbrowser.open(REPO_URL)

    def _load_versions(self):
        def worker():
            ollama_version = detect_ollama_version()
            if ollama_version:
                message = f"App v{__version__} | {ollama_version}"
            else:
                message = f"App v{__version__} | Ollama version: unavailable"
            self.root.after(0, lambda: self.version_var.set(message))

        threading.Thread(target=worker, daemon=True).start()

    def append_log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def clear_log(self):
        if self.worker and self.worker.is_alive():
            return
        self.log_text.delete("1.0", "end")

    def _set_running(self, running, status):
        self.status_var.set(status)
        action_state = "disabled" if running else "normal"
        stop_state = "normal" if running else "disabled"
        self.update_button.configure(state=action_state)
        self.test_button.configure(state=action_state)
        self.stop_button.configure(state=stop_state)
        self.test_stop_button.configure(state=stop_state)

    def _drain_log_queue(self):
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item["kind"] == "log":
                self.append_log(item["message"])
            elif item["kind"] == "done":
                self.report_var.set(f"Report: {Path(self.report_path_var.get()).resolve()}")
                self._set_running(False, item["status"])
                self.active_job_var.set("No job running")
                self.stop_event.clear()

        self.root.after(100, self._drain_log_queue)

    def _run_worker(self, label, target):
        if self.worker and self.worker.is_alive():
            return

        self.stop_event.clear()
        self._set_running(True, f"Running {label}...")
        self.active_job_var.set(f"{label} job in progress")
        self.append_log(f"=== {label} ===")

        def emit(message):
            self.log_queue.put({"kind": "log", "message": message})

        def runner():
            code = target(emit, lambda: self.stop_event.is_set())
            status = f"{label} finished with exit code {code}"
            self.log_queue.put({"kind": "done", "status": status})

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def _parse_timeout(self):
        timeout_text = self.timeout_var.get().strip()
        if not timeout_text:
            return None, "Timeout is required."

        try:
            timeout_value = int(timeout_text)
        except ValueError:
            return None, "Timeout must be an integer number of seconds."

        if timeout_value <= 0:
            return None, "Timeout must be greater than zero."

        return timeout_value, None

    def _parse_vram_override(self):
        vram_text = self.vram_override_var.get().strip()
        if not vram_text:
            return None, None

        try:
            vram_value = int(vram_text)
        except ValueError:
            return None, "Manual VRAM must be an integer MiB value."

        if vram_value <= 0:
            return None, "Manual VRAM must be greater than zero."

        return vram_value, None

    def _selected_report_path(self):
        report_path = self.report_path_var.get().strip()
        if not report_path:
            return None, "Report path is required."

        return str(Path(report_path).expanduser()), None

    def _selected_api_base_url(self):
        try:
            return normalize_ollama_api_base_url(self.api_base_url_var.get()), None
        except ValueError as exc:
            return None, str(exc)

    def run_update(self):
        self._run_worker(
            "Update",
            lambda emit, stop_requested: update_main(
                ["tz-ollama-utils-update"],
                emit,
                stop_requested=stop_requested,
            ),
        )

    def choose_report_path(self):
        path = self.filedialog.asksaveasfilename(
            title="Choose YAML report path",
            defaultextension=".yaml",
            initialfile=Path(self.report_path_var.get()).name,
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.report_path_var.set(path)

    def stop_run(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Stopping...")
            self.active_job_var.set("Waiting for clean shutdown")
            self.append_log("Stop requested. Waiting for the current step to finish cleanly...")

    def run_test(self):
        timeout, timeout_error = self._parse_timeout()
        if timeout_error:
            self.append_log(timeout_error)
            self.status_var.set("Invalid timeout")
            return

        vram_override, vram_error = self._parse_vram_override()
        if vram_error:
            self.append_log(vram_error)
            self.status_var.set("Invalid VRAM override")
            return

        report_path, report_error = self._selected_report_path()
        if report_error:
            self.append_log(report_error)
            self.status_var.set("Invalid report path")
            return

        api_base_url, api_base_url_error = self._selected_api_base_url()
        if api_base_url_error:
            self.append_log(api_base_url_error)
            self.status_var.set("Invalid API URL")
            return

        argv = ["tz-ollama-utils-test", str(timeout)]
        if self.ignore_size_var.get():
            argv.append("--ignore-size")
        if vram_override is not None:
            argv.extend(["--vram-mib", str(vram_override)])
        argv.extend(["--api-base-url", api_base_url])
        argv.extend(["--report-path", report_path])

        self._run_worker(
            "Test",
            lambda emit, stop_requested: test_main(
                argv,
                emit,
                stop_requested=stop_requested,
            ),
        )


def main():
    try:
        import tkinter as tk
        from tkinter import ttk
    except ModuleNotFoundError:
        print(
            "Unable to launch the GUI because this Python installation does not include tkinter.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    _configure_windows_app_id()
    root = tk.Tk()
    OllamaUtilsApp(root, tk, ttk)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
