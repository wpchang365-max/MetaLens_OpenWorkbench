from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class RoundedButton(tk.Canvas):
    def __init__(self, parent, text: str, command=None, *, accent: bool = False, width: int = 132, height: int = 34, anchor: str = "center"):
        bg = "#0F6CBD" if accent else "#FFFFFF"
        fg = "#FFFFFF" if accent else "#1B1B1F"
        try:
            parent_bg = parent.cget("background")
        except tk.TclError:
            style_name = parent.cget("style") if isinstance(parent, ttk.Widget) else ""
            parent_bg = "#FFFFFF" if style_name == "Card.TFrame" else "#F5F7FA"
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, bd=0, cursor="hand2")
        self._text, self._command, self._accent, self._state, self._anchor = text, command, accent, "normal", anchor
        self._fill, self._fg = bg, fg
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda _event: self._draw(hover=True))
        self.bind("<Leave>", lambda _event: self._draw())
        self._draw()

    def _rounded(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius, x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2, x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def _draw(self, hover=False):
        self.delete("all")
        w, h = max(8, self.winfo_width()), max(8, self.winfo_height())
        if self._state == "disabled":
            fill, fg = "#D9DEE5", "#8A929E"
        elif hover:
            fill, fg = ("#115EA3", "#FFFFFF") if self._accent else ("#E8F1FA", self._fg)
        else:
            fill, fg = self._fill, self._fg
        self._rounded(1, 1, w - 1, h - 1, 9, fill=fill, outline="#D5DAE0" if not self._accent else fill)
        x = 13 if self._anchor == "w" else w / 2
        self.create_text(x, h / 2, text=self._text, fill=fg, anchor=self._anchor, font=("Segoe UI", 9, "bold" if self._accent else "normal"))

    def _click(self, _event):
        if self._state != "disabled" and self._command:
            self._command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "text" in kwargs: self._text = kwargs.pop("text")
        if "state" in kwargs: self._state = kwargs.pop("state")
        if "command" in kwargs: self._command = kwargs.pop("command")
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    def cget(self, key):
        if key == "text": return self._text
        if key == "state": return self._state
        return super().cget(key)

    def __getitem__(self, key):
        return self.cget(key)


class FluentProgress(tk.Canvas):
    def __init__(self, parent, *, height: int = 8, accent: str = "#0F6CBD", track: str = "#E2E7ED"):
        try:
            bg = parent.cget("background")
        except tk.TclError:
            bg = "#FFFFFF" if isinstance(parent, ttk.Widget) and parent.cget("style") == "Card.TFrame" else "#F5F7FA"
        super().__init__(parent, height=height, bg=bg, highlightthickness=0, bd=0)
        self.value, self.maximum, self.accent, self.track = 0.0, 100.0, accent, track
        self.bind("<Configure>", lambda _event: self._draw())

    def _draw(self):
        self.delete("all")
        w, h = max(4, self.winfo_width()), max(4, self.winfo_height())
        radius = h / 2
        self.create_rectangle(radius, 1, w - radius, h - 1, fill=self.track, outline="")
        self.create_oval(1, 1, h - 1, h - 1, fill=self.track, outline="")
        self.create_oval(w - h + 1, 1, w - 1, h - 1, fill=self.track, outline="")
        filled = max(0, min(w, w * self.value / max(self.maximum, 1e-9)))
        if filled > 1:
            self.create_rectangle(radius, 1, max(radius, filled - radius), h - 1, fill=self.accent, outline="")
            self.create_oval(1, 1, h - 1, h - 1, fill=self.accent, outline="")
            if filled >= h:
                self.create_oval(filled - h + 1, 1, filled - 1, h - 1, fill=self.accent, outline="")

    def configure(self, cnf=None, **kwargs):
        if cnf: kwargs.update(cnf)
        if "value" in kwargs: self.value = float(kwargs.pop("value"))
        if "maximum" in kwargs: self.maximum = float(kwargs.pop("maximum"))
        if kwargs: super().configure(**kwargs)
        self._draw()

    config = configure

    def __setitem__(self, key, value):
        self.configure(**{key: value})

    def __getitem__(self, key):
        if key == "value": return self.value
        if key == "maximum": return self.maximum
        return super().__getitem__(key)

    def start(self, interval=50):
        self._pulse = True
        def tick():
            if getattr(self, "_pulse", False):
                self.value = (self.value + 4) % 100
                self._draw(); self.after(interval, tick)
        tick()

    def stop(self):
        self._pulse = False
        self.value = 0; self._draw()


class FluentComboBox(tk.Canvas):
    """Rounded Windows 11-style readonly selection control."""

    def __init__(self, parent, variable: tk.StringVar, *, values=(), width: int = 168, height: int = 34):
        try:
            parent_bg = parent.cget("background")
        except tk.TclError:
            parent_bg = "#FFFFFF"
        super().__init__(parent, width=width, height=height, bg=parent_bg, highlightthickness=0, bd=0, cursor="hand2", takefocus=True)
        self.variable = variable
        self._values = list(values)
        self._popup = None
        self._hover = False
        self.bind("<Configure>", lambda _event: self._draw())
        self.bind("<Enter>", lambda _event: self._set_hover(True))
        self.bind("<Leave>", lambda _event: self._set_hover(False))
        self.bind("<Button-1>", self._toggle_popup)
        self.bind("<space>", self._toggle_popup)
        self.bind("<Return>", self._toggle_popup)
        self.variable.trace_add("write", lambda *_args: self._draw())
        self._draw()

    def _rounded(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius, x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2, x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def _set_hover(self, value: bool):
        self._hover = value
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(40, self.winfo_width()), max(28, self.winfo_height())
        outline = "#0F6CBD" if self._hover or self._popup is not None else "#C9CED6"
        self._rounded(1, 1, w - 1, h - 1, 8, fill="#FFFFFF", outline=outline, width=1)
        self.create_text(12, h / 2, text=self.variable.get(), anchor="w", fill="#1B1B1F", font=("Segoe UI", 9))
        self.create_line(w - 20, h / 2 - 2, w - 15, h / 2 + 3, w - 10, h / 2 - 2, fill="#44505E", width=1.5, smooth=True)

    def _toggle_popup(self, _event=None):
        if self._popup is not None:
            self._close_popup()
            return "break"
        popup = tk.Toplevel(self)
        self._popup = popup
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(bg="#D6DAE0")
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 3
        popup.geometry(f"{max(self.winfo_width(), 168)}x{max(34, len(self._values) * 32 + 8)}+{x}+{y}")
        body = tk.Frame(popup, bg="#FFFFFF", padx=4, pady=4, highlightbackground="#D6DAE0", highlightthickness=1)
        body.pack(fill="both", expand=True)
        for value in self._values:
            row = tk.Label(body, text=value, bg="#FFFFFF", fg="#1B1B1F", anchor="w", padx=10, pady=6, font=("Segoe UI", 9), cursor="hand2")
            row.pack(fill="x")
            row.bind("<Enter>", lambda event: event.widget.configure(bg="#E8F1FA"))
            row.bind("<Leave>", lambda event: event.widget.configure(bg="#FFFFFF"))
            row.bind("<Button-1>", lambda _event, selected=value: self._select(selected))
        popup.bind("<Escape>", lambda _event: self._close_popup())
        popup.bind("<FocusOut>", lambda _event: self.after(20, self._close_if_unfocused))
        popup.focus_force()
        self._draw()
        return "break"

    def _select(self, value: str):
        self.variable.set(value)
        self._close_popup()

    def _close_if_unfocused(self):
        if self._popup is not None and self._popup.focus_displayof() is None:
            self._close_popup()

    def _close_popup(self):
        if self._popup is not None:
            try:
                self._popup.destroy()
            except tk.TclError:
                pass
            self._popup = None
            self._draw()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "values" in kwargs:
            self._values = list(kwargs.pop("values"))
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    config = configure

    def cget(self, key):
        if key == "values":
            return tuple(self._values)
        return super().cget(key)


class FluentLanguageSwitch(tk.Canvas):
    """Compact two-segment language selector for the navigation footer."""

    def __init__(self, parent, language="zh", command=None, *, width=190, height=38):
        super().__init__(parent, width=width, height=height, bg="#EEF2F6", highlightthickness=0, bd=0, cursor="hand2")
        self.language, self.command = language, command
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Button-1>", self._click)
        self._draw()

    def _rounded(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1+radius,y1,x2-radius,y1,x2,y1,x2,y1+radius,x2,y2-radius,x2,y2,x2-radius,y2,x1+radius,y2,x1,y2,x1,y2-radius,x1,y1+radius,x1,y1]
        return self.create_polygon(points, smooth=True, splinesteps=24, **kwargs)

    def _draw(self):
        self.delete("all"); w,h=max(100,self.winfo_width()),max(30,self.winfo_height())
        self._rounded(1,1,w-1,h-1,9,fill="#FFFFFF",outline="#D5DAE0")
        half=w/2
        if self.language == "zh":
            self._rounded(3,3,half,h-3,7,fill="#0F6CBD",outline="")
        else:
            self._rounded(half,3,w-3,h-3,7,fill="#0F6CBD",outline="")
        self.create_text(half/2,h/2,text="中文",fill="#FFFFFF" if self.language=="zh" else "#1B1B1F",font=("Microsoft YaHei UI",9,"bold"))
        self.create_text(half+half/2,h/2,text="English",fill="#FFFFFF" if self.language=="en" else "#1B1B1F",font=("Segoe UI",9,"bold"))

    def _click(self, event):
        selected = "zh" if event.x < self.winfo_width()/2 else "en"
        if selected != self.language:
            self.language = selected; self._draw()
            if self.command: self.command(selected)

    def set_language(self, language):
        self.language = language; self._draw()


class FluentScrollFrame(ttk.Frame):
    """Scrollable frame with a thin Windows 11-like hover-expanding thumb."""

    def __init__(self, parent, *, background: str = "#F5F7FA"):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=background)
        self.content = ttk.Frame(self.canvas)
        self.track = tk.Canvas(self, width=6, highlightthickness=0, bg=background, cursor="arrow")
        self.thumb = self.track.create_rectangle(2, 0, 4, 40, fill="#8A94A6", outline="")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.track.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._layout)
        self.canvas.bind("<Configure>", self._layout)
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")
        self.track.bind("<Enter>", lambda _e: self._set_width(12))
        self.track.bind("<Leave>", lambda _e: self._set_width(6))
        self.track.bind("<Button-1>", self._jump)
        self.track.bind("<B1-Motion>", self._drag)
        self._drag_offset = 0

    def _set_width(self, width: int):
        self.track.configure(width=width)
        self.track.coords(self.thumb, 2, self.track.coords(self.thumb)[1], width - 2, self.track.coords(self.thumb)[3])

    def _layout(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfigure(self.window, width=self.canvas.winfo_width())
        self._sync_thumb()

    def _wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        if not self._owns_pointer(widget):
            return
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return "break"
        steps = int(-event.delta / 120)
        if steps == 0 and event.delta:
            steps = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(steps, "units")
        self._sync_thumb()
        return "break"

    def _owns_pointer(self, widget) -> bool:
        """Return true only for the closest scroll frame below the pointer."""
        while widget is not None:
            if isinstance(widget, FluentScrollFrame):
                return widget is self
            widget = getattr(widget, "master", None)
        return False

    def _sync_thumb(self):
        first, last = self.canvas.yview()
        height = max(1, self.track.winfo_height())
        top, bottom = first * height, max(first * height + 32, last * height)
        width = int(self.track.cget("width"))
        self.track.coords(self.thumb, 2, top, width - 2, min(height, bottom))

    def _jump(self, event):
        y1, y2 = self.track.coords(self.thumb)[1::2]
        if y1 <= event.y <= y2:
            self._drag_offset = event.y - y1
        else:
            self.canvas.yview_moveto(event.y / max(1, self.track.winfo_height()))
            self._sync_thumb()

    def _drag(self, event):
        height = max(1, self.track.winfo_height())
        thumb_height = self.track.coords(self.thumb)[3] - self.track.coords(self.thumb)[1]
        fraction = (event.y - self._drag_offset) / max(1, height - thumb_height)
        self.canvas.yview_moveto(max(0.0, min(1.0, fraction)))
        self._sync_thumb()


class ToggleCheck(ttk.Frame):
    def __init__(self, parent, text: str, variable: tk.BooleanVar, command=None):
        super().__init__(parent)
        self.variable = variable
        self.command = command
        self.button = tk.Button(self, width=2, relief="flat", bd=0, command=self.toggle, font=("Segoe UI Symbol", 11), cursor="hand2")
        self.label = ttk.Label(self, text=text)
        self.button.pack(side="left")
        self.label.pack(side="left", padx=(7, 0))
        self.label.bind("<Button-1>", lambda _e: self.toggle())
        try:
            self.variable.trace_add("write", lambda *_args: self.refresh())
        except AttributeError:
            self.variable.trace("w", lambda *_args: self.refresh())
        self.refresh()

    def toggle(self):
        self.variable.set(not self.variable.get())
        self.refresh()
        if self.command:
            self.command()

    def refresh(self):
        enabled = self.variable.get()
        self.button.configure(text="✓" if enabled else "", bg="#0F6CBD" if enabled else "#FFFFFF", fg="#FFFFFF", activebackground="#115EA3")
