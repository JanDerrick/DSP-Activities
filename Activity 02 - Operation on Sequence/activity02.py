
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ---------------- Time-index operations ----------------
def delay(n, y, k):
    """y[n] = x[n - k]: indices move right by k, values unchanged."""
    return n + k, y

def advance(n, y, k):
    """y[n] = x[n + k]: indices move left by k, values unchanged."""
    return n - k, y

def fold(n, y):
    """y[n] = x[-n]: reflect about n = 0."""
    return (-n)[::-1], y[::-1]

# ---------------- Amplitude operations ----------------
def scale(n, y, a):
    """y[n] = a * x[n]."""
    return n, a * y

def align(n1, y1, n2, y2):
    """Place two sequences on a common index range; missing samples are zero."""
    lo, hi = min(n1[0], n2[0]), max(n1[-1], n2[-1])
    n = np.arange(lo, hi + 1)
    a, b = np.zeros(len(n)), np.zeros(len(n))
    a[n1 - lo] = y1
    b[n2 - lo] = y2
    return n, a, b

def add(n1, y1, n2, y2):
    """y[n] = x1[n] + x2[n]."""
    n, a, b = align(n1, y1, n2, y2)
    return n, a + b

def multiply(n1, y1, n2, y2):
    """y[n] = x1[n] * x2[n]; zero samples at the edges are dropped."""
    n, a, b = align(n1, y1, n2, y2)
    y = a * b
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return np.array([0]), np.array([0.0])
    return n[nz[0]:nz[-1] + 1], y[nz[0]:nz[-1] + 1]

# ---------------- Composite operation y[n] = x[-n + k] ----------------
def composite_method1(n, y, k):
    """Shift then fold. Returns (intermediate v[n], result y[n])."""
    v = advance(n, y, k)        # v[n] = x[n + k]
    return v, fold(*v)          # y[n] = v[-n]

def composite_method2(n, y, k):
    """Fold then shift. Returns (intermediate v[n], result y[n])."""
    v = fold(n, y)              # v[n] = x[-n]
    return v, delay(*v, k)      # y[n] = v[n - k]

# ---------------- Block diagram elements ----------------
def unit_delay(n, y):
    return delay(n, y, 1)                # z^-1: y[n] = x[n - 1]

def unit_advance(n, y):
    return advance(n, y, 1)              # z: y[n] = x[n + 1]

def adder(s1, s2):
    return add(*s1, *s2)                 # y[n] = x1[n] + x2[n]

def constant_multiplier(a, s):
    return scale(*s, a)                  # y[n] = a * x[n]

def signal_multiplier(s1, s2):
    return multiply(*s1, *s2)            # y[n] = x1[n] * x2[n]

# ---------------- Computation and text report ----------------
def compute_all(x, n0, k, a, x1, n01, x2, n02):
    """Run every operation and return the results as (n, y) pairs."""
    x = np.asarray(x, float)
    s = (np.arange(n0, n0 + len(x)), x)
    s1 = (np.arange(n01, n01 + len(x1)), np.asarray(x1, float))
    s2 = (np.arange(n02, n02 + len(x2)), np.asarray(x2, float))
    m1_mid, m1 = composite_method1(*s, k)
    m2_mid, m2 = composite_method2(*s, k)

    return {
        # single-signal operations
        "original": s,
        "scale": scale(*s, a),
        "delay": delay(*s, k),
        "advance": advance(*s, k),
        "reversal": fold(*s),
        "m1_mid": m1_mid, "m1": m1,
        "m2_mid": m2_mid, "m2": m2,
        # two-signal operations
        "x1": s1, "x2": s2,
        "sum": add(*s1, *s2),
        "product": multiply(*s1, *s2),
        # block diagram elements
        "unit_delay": unit_delay(*s),
        "unit_advance": unit_advance(*s),
        "adder": adder(s, unit_delay(*s)),
        "const_mult": constant_multiplier(a, s),
        "sig_mult": signal_multiplier(s, unit_delay(*s)),
        "energy": float(np.sum(np.abs(x) ** 2)),
    }

def fmt(label, seq):
    """Format a sequence like the slides, with ^ marking n = 0."""
    n, y = seq
    items = [f"{v:g}" for v in y]
    prefix = f"  {label}[n] = {{"
    lines = [prefix + ", ".join(items) + f"}}   for n = {int(n[0])}..{int(n[-1])}"]
    if n[0] <= 0 <= n[-1]:
        pos = len(prefix) + sum(len(t) + 2 for t in items[:int(-n[0])])
        lines.append(" " * pos + "^ (n = 0)")
    else:
        lines.append("  (n = 0 is outside the listed samples)")
    return "\n".join(lines)

def build_report(r, x, n0, k, a):
    """Build the numeric results shown in the text box."""
    out = [f"Original: x[n] over n = {n0}..{n0 + len(x) - 1}", fmt("x", r["original"]), ""]
    out += [f"1) Scalar multiplication (a = {a:g}): y[n] = a*x[n]", fmt("y", r["scale"]), ""]
    out += [f"2) Time delay: y[n] = x[n - {k}]", fmt("y", r["delay"]), ""]
    out += [f"3) Time advance: y[n] = x[n + {k}]", fmt("y", r["advance"]), ""]
    out += ["4) Time reversal: y[n] = x[-n]", fmt("y", r["reversal"]), ""]
    out += [f"5) Composite Method 1 (shift then fold): y[n] = x[-n + {k}]",
            fmt("v", r["m1_mid"]) + "    <- v[n] = x[n + k]", fmt("y", r["m1"]), ""]
    out += [f"6) Composite Method 2 (fold then shift): y[n] = x[-n + {k}]",
            fmt("v", r["m2_mid"]) + "    <- v[n] = x[-n]", fmt("y", r["m2"])]
    same = np.array_equal(r["m1"][0], r["m2"][0]) and np.allclose(r["m1"][1], r["m2"][1])
    out += [f"  Method 1 and Method 2: {'SAME result (confirmed)' if same else 'MISMATCH - check inputs'}", ""]
    out += ["7) Signal addition: y[n] = x1[n] + x2[n]",
            fmt("x1", r["x1"]), fmt("x2", r["x2"]), fmt("y", r["sum"]), ""]
    out += ["8) Signal multiplication: y[n] = x1[n] * x2[n]", fmt("y", r["product"]), ""]
    out.append(f"Energy of x[n]: E = sum |x[n]|^2 = {r['energy']:g}")
    return "\n".join(out)

# ---------------- Tkinter application ----------------
class SequenceApp:
    # Default inputs: the worked examples from the slides
    DEFAULTS = {"x": "1, 2, 3, 4, 5", "n0": "-2", "k": "2", "a": "2",
                "x1": "1, 2, 3", "n01": "-1", "x2": "4, 5, 6", "n02": "0"}

    def __init__(self, root):
        self.root = root
        self.root.title("Discrete-Time Sequence Operations")
        self.root.geometry("1280x900")
        self._build_inputs()
        self._build_output()
        self._build_tabs()
        self.compute_and_plot()

    # ----- layout -----
    def _entry(self, parent, label, key, width, row, col):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
        e = ttk.Entry(parent, width=width)
        e.insert(0, self.DEFAULTS[key])
        e.grid(row=row, column=col + 1, sticky="w", padx=4, pady=3)
        self.entries[key] = e

    def _build_inputs(self):
        frame = ttk.LabelFrame(self.root, text="Input")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        self.entries = {}

        # Row 0: single-signal operations
        self._entry(frame, "x[n] values:", "x", 26, 0, 0)
        self._entry(frame, "start index n0:", "n0", 6, 0, 2)
        self._entry(frame, "shift k:", "k", 6, 0, 4)
        self._entry(frame, "gain a:", "a", 6, 0, 6)
        # Row 1: two-signal operations
        self._entry(frame, "x1[n] values:", "x1", 26, 1, 0)
        self._entry(frame, "x1 start n0:", "n01", 6, 1, 2)
        self._entry(frame, "x2[n] values:", "x2", 14, 1, 4)
        self._entry(frame, "x2 start n0:", "n02", 6, 1, 6)

        ttk.Button(frame, text="Compute & Plot", command=self.compute_and_plot).grid(
            row=0, column=8, padx=12, pady=3)
        ttk.Button(frame, text="Reset Example", command=self.reset).grid(
            row=1, column=8, padx=12, pady=3)

    def _build_output(self):
        frame = ttk.LabelFrame(self.root, text="Numeric Results")
        frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)
        self.text = ScrolledText(frame, height=12, wrap=tk.NONE, font=("Consolas", 10))
        self.text.pack(fill=tk.X, padx=5, pady=5)

    def _build_tabs(self):
        tabs = ttk.Notebook(self.root)
        tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.figs = {}
        specs = [("single", "Single-signal operations", (2, 4)),
                ("double", "Two-signal operations", (2, 2)),
                ("blocks", "Block diagram elements", (2, 3))]
        for key, title, shape in specs:
            page = ttk.Frame(tabs)
            tabs.add(page, text=title)
            fig = Figure(figsize=(12, 5.2), dpi=100)
            axes = fig.subplots(*shape)
            canvas = FigureCanvasTkAgg(fig, master=page)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            bar = ttk.Frame(page)
            bar.pack(side=tk.TOP, fill=tk.X)
            NavigationToolbar2Tk(canvas, bar).update()
            self.figs[key] = (fig, canvas, list(np.atleast_1d(axes).flat))

    # ----- actions -----
    def reset(self):
        for key, e in self.entries.items():
            e.delete(0, tk.END)
            e.insert(0, self.DEFAULTS[key])
        self.compute_and_plot()

    def _values(self, key, name):
        """Parse a comma-separated list of numbers."""
        try:
            v = [float(t) for t in self.entries[key].get().split(",") if t.strip()]
        except ValueError:
            raise ValueError(f"{name} must be numbers separated by commas, e.g. 1, 2, 3")
        if not v:
            raise ValueError(f"Please enter at least one value for {name}.")
        return v

    def _number(self, key, name, kind):
        """Parse a single int or float."""
        try:
            return kind(self.entries[key].get().strip())
        except ValueError:
            raise ValueError(f"{name} must be {'an integer' if kind is int else 'a number'}.")

    def compute_and_plot(self):
        try:
            x = self._values("x", "x[n]")
            x1 = self._values("x1", "x1[n]")
            x2 = self._values("x2", "x2[n]")
            n0 = self._number("n0", "n0", int)
            n01 = self._number("n01", "x1 start n0", int)
            n02 = self._number("n02", "x2 start n0", int)
            k = self._number("k", "Shift k", int)
            a = self._number("a", "Gain a", float)
        except ValueError as err:
            messagebox.showerror("Invalid input", str(err))
            return

        r = compute_all(x, n0, k, a, x1, n01, x2, n02)
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, build_report(r, x, n0, k, a))

        self._draw("single", [
            (r["original"], "Original: x[n]"),
            (r["scale"], f"Scalar mult: y[n] = {a:g}·x[n]"),
            (r["delay"], f"Time delay: y[n] = x[n - {k}]"),
            (r["advance"], f"Time advance: y[n] = x[n + {k}]"),
            (r["reversal"], "Time reversal: y[n] = x[-n]"),
            (r["m1"], f"Composite M1 (shift→fold): x[-n + {k}]"),
            (r["m2"], f"Composite M2 (fold→shift): x[-n + {k}]"),
        ])
        self._draw("double", [
            (r["x1"], "x1[n]"),
            (r["x2"], "x2[n]"),
            (r["sum"], "Addition: x1[n] + x2[n]"),
            (r["product"], "Multiplication: x1[n] · x2[n]"),
        ])
        self._draw("blocks", [
            (r["original"], "Input x[n]"),
            (r["unit_delay"], "Unit delay z⁻¹: x[n - 1]"),
            (r["unit_advance"], "Unit advance z: x[n + 1]"),
            (r["adder"], "Adder: x[n] + x[n - 1]"),
            (r["const_mult"], f"Constant multiplier: {a:g}·x[n]"),
            (r["sig_mult"], "Signal multiplier: x[n]·x[n - 1]"),
        ])

    def _draw(self, key, panels):
        """Draw stem plots on one tab using a shared index range."""
        fig, canvas, axes = self.figs[key]
        lo = min(int(n[0]) for (n, _), _ in panels) - 1
        hi = max(int(n[-1]) for (n, _), _ in panels) + 1
        for ax, ((n, y), title) in zip(axes, panels):
            ax.clear()
            ax.stem(n, y, basefmt=" ")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("n", fontsize=8)
            ax.set_ylabel("value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlim(lo - 0.5, hi + 0.5)
            ax.set_xticks(range(lo, hi + 1))
            ax.grid(True, alpha=0.4)
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", lw=0.6)
        for ax in axes[len(panels):]:
            ax.axis("off")                  # hide unused panels
        fig.tight_layout(pad=2.0)
        canvas.draw()

def main():
    root = tk.Tk()
    SequenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()