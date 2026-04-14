import os
import tkinter as tk
from tkinter import messagebox, ttk

import config
import database
import utils

utils.setup_logging()


class AttendanceViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Attendance Viewer")
        self.root.geometry("920x620")
        self.root.minsize(840, 540)

        database.init_db()
        database.export_all_dates_csv()

        self.selected_date = tk.StringVar()
        self.info_var = tk.StringVar(value="")

        self._configure_style()
        self._build_ui()
        self.refresh_dates()

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1d3557")
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#5f6d7a")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 10))

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Attendance Records", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="Select a date to view attendance. CSV files are generated per date in data/processed.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Date:").pack(side="left")
        self.date_combo = ttk.Combobox(top, textvariable=self.selected_date, state="readonly", width=18)
        self.date_combo.pack(side="left", padx=(8, 8))
        self.date_combo.bind("<<ComboboxSelected>>", lambda _e: self.load_selected_date())

        ttk.Button(top, text="Refresh", command=self.refresh_dates).pack(side="left")
        ttk.Button(top, text="Export Selected Date CSV", command=self.export_selected_date).pack(side="left", padx=(8, 0))

        self.table = ttk.Treeview(
            frame,
            columns=("name", "date", "time", "status"),
            show="headings",
            height=20,
        )
        self.table.heading("name", text="Name")
        self.table.heading("date", text="Date")
        self.table.heading("time", text="Time")
        self.table.heading("status", text="Status")
        self.table.column("name", width=220)
        self.table.column("date", width=110)
        self.table.column("time", width=110)
        self.table.column("status", width=100)
        self.table.pack(side="left", fill="both", expand=True)

        yscroll = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side="right", fill="y")

        ttk.Label(frame, textvariable=self.info_var, style="Sub.TLabel").pack(anchor="w", pady=(8, 0))

    def refresh_dates(self):
        dates = database.get_all_available_attendance_dates()
        self.date_combo["values"] = dates

        if not dates:
            self.selected_date.set("")
            self._render_rows([])
            self.info_var.set("No attendance data found yet.")
            return

        if self.selected_date.get() not in dates:
            self.selected_date.set(dates[0])

        self.load_selected_date()

    def load_selected_date(self):
        date_str = self.selected_date.get().strip()
        if not date_str:
            self._render_rows([])
            self.info_var.set("Select a date.")
            return

        rows = database.get_attendance_for_date(date_str)
        if not rows:
            rows = database.load_attendance_csv_by_date(date_str)
        self._render_rows(rows)
        self.info_var.set(f"{len(rows)} record(s) for {date_str}")

    def export_selected_date(self):
        date_str = self.selected_date.get().strip()
        if not date_str:
            messagebox.showwarning("Export", "Please select a date first.")
            return

        filepath = database.export_date_csv(date_str)
        messagebox.showinfo("Export", f"CSV created:\n{filepath}")

    def _render_rows(self, rows):
        for item in self.table.get_children():
            self.table.delete(item)
        for row in rows:
            self.table.insert(
                "",
                "end",
                values=(row["name"], row["date"], row["time"], row["status"]),
            )


def main():
    root = tk.Tk()
    app = AttendanceViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
