import sys
import queue
import threading
from pathlib import Path

from .test_models import FAILURE_LOG
from .test_models import main as test_main
from .update_models import main as update_main


class OllamaMaintenanceApp:
    def __init__(self, root, tk_module, ttk_module):
        self.tk = tk_module
        self.ttk = ttk_module
        self.root = root
        self.root.title("Ollama Maintenance")
        self.root.geometry("980x720")
        self.root.minsize(820, 560)

        self.log_queue = queue.Queue()
        self.worker = None

        self.timeout_var = self.tk.StringVar(value="300")
        self.ignore_size_var = self.tk.BooleanVar(value=False)
        self.status_var = self.tk.StringVar(value="Idle")
        self.report_var = self.tk.StringVar(value=f"Report: {FAILURE_LOG.resolve()}")

        self._build_ui()
        self.root.after(100, self._drain_log_queue)

    def _build_ui(self):
        ttk = self.ttk
        tk = self.tk
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=16)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(7, weight=1)

        ttk.Label(controls, text="Per-model timeout (seconds)").grid(
            row=0, column=0, sticky="w"
        )
        timeout_entry = ttk.Entry(controls, textvariable=self.timeout_var, width=10)
        timeout_entry.grid(row=0, column=1, padx=(8, 16), sticky="w")

        ignore_size = ttk.Checkbutton(
            controls,
            text="Ignore VRAM size filter",
            variable=self.ignore_size_var,
        )
        ignore_size.grid(row=0, column=2, padx=(0, 16), sticky="w")

        self.update_button = ttk.Button(
            controls,
            text="Update Installed Models",
            command=self.run_update,
        )
        self.update_button.grid(row=0, column=3, padx=(0, 8), sticky="ew")

        self.test_button = ttk.Button(
            controls,
            text="Test and Inventory Models",
            command=self.run_test,
        )
        self.test_button.grid(row=0, column=4, padx=(0, 8), sticky="ew")

        self.clear_button = ttk.Button(controls, text="Clear Log", command=self.clear_log)
        self.clear_button.grid(row=0, column=5, padx=(0, 8), sticky="ew")

        ttk.Label(controls, textvariable=self.status_var).grid(
            row=0, column=7, sticky="e"
        )

        log_frame = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            bg="#101418",
            fg="#E6EDF3",
            insertbackground="#E6EDF3",
            padx=12,
            pady=12,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(self.root, padding=(16, 0, 16, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)

        ttk.Label(footer, textvariable=self.report_var).grid(row=0, column=0, sticky="w")
        ttk.Label(
            footer,
            text="Requires local Ollama CLI/API. Generated binaries do not bundle Ollama itself.",
        ).grid(row=1, column=0, pady=(6, 0), sticky="w")

    def append_log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def clear_log(self):
        if self.worker and self.worker.is_alive():
            return
        self.log_text.delete("1.0", "end")

    def _set_running(self, running, status):
        self.status_var.set(status)
        state = "disabled" if running else "normal"
        self.update_button.configure(state=state)
        self.test_button.configure(state=state)

    def _drain_log_queue(self):
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item["kind"] == "log":
                self.append_log(item["message"])
            elif item["kind"] == "done":
                self.report_var.set(f"Report: {Path(FAILURE_LOG).resolve()}")
                self._set_running(False, item["status"])

        self.root.after(100, self._drain_log_queue)

    def _run_worker(self, label, target):
        if self.worker and self.worker.is_alive():
            return

        self._set_running(True, f"Running {label}...")
        self.append_log(f"=== {label} ===")

        def emit(message):
            self.log_queue.put({"kind": "log", "message": message})

        def runner():
            code = target(emit)
            status = f"{label} finished with exit code {code}"
            self.log_queue.put({"kind": "done", "status": status})

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def run_update(self):
        self._run_worker("Update", lambda emit: update_main(["ollama-maintenance-update"], emit))

    def run_test(self):
        timeout = self.timeout_var.get().strip()
        argv = ["ollama-maintenance-test"]
        if timeout:
            argv.append(timeout)
        if self.ignore_size_var.get():
            argv.append("--ignore-size")

        self._run_worker("Test", lambda emit: test_main(argv, emit))


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

    root = tk.Tk()
    app = OllamaMaintenanceApp(root, tk, ttk)
    del app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
