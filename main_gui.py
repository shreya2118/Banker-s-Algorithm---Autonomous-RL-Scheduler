import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from rl_brain import train_agent_live  # Import our trainer
from compare_models import BenchmarkArena

class RLBankersApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Banker's Algorithm - Autonomous RL Scheduler")
        self.geometry("1000x800")
        
        # Default State
        self.n_var = tk.IntVar(value=5)
        self.m_var = tk.IntVar(value=3)
        self.entries_alloc = []
        self.entries_max = []
        self.entries_avail = []
        
        # RL State
        self.q_table = None
        self.current_alloc = None
        self.current_avail = None
        self.finished_mask = []
        
        self._setup_ui()
        self._generate_grid()

    def _setup_ui(self):
        # --- CONTROL PANEL ---
        top_frame = ttk.LabelFrame(self, text="Configuration")
        top_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(top_frame, text="Processes:").pack(side="left", padx=5)
        ttk.Spinbox(top_frame, from_=2, to=10, textvariable=self.n_var, width=5, command=self._generate_grid).pack(side="left")
        
        ttk.Label(top_frame, text="Resources:").pack(side="left", padx=5)
        ttk.Spinbox(top_frame, from_=1, to=10, textvariable=self.m_var, width=5, command=self._generate_grid).pack(side="left")
        
        ttk.Button(top_frame, text="Reset / Rebuild Tables", command=self._generate_grid).pack(side="left", padx=15)
        
        # --- MATRICES INPUT ---
        self.matrix_frame = ttk.Frame(self)
        self.matrix_frame.pack(pady=10)
        
        # --- ACTION PANEL ---
        action_frame = ttk.Frame(self)
        action_frame.pack(pady=10)
        
        self.status_label = ttk.Label(action_frame, text="Status: Ready for input", font=("Arial", 12, "bold"), foreground="gray")
        self.status_label.pack(pady=5)
        
        self.btn_run = ttk.Button(action_frame, text="🤖 TRAIN AGENT & AUTO-PILOT", command=self.start_procedure)
        self.btn_run.pack(ipadx=10, ipady=5)

        # --- LOG / VISUALIZATION ---
        log_frame = ttk.LabelFrame(self, text="Real-Time Execution Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True)

        self.btn_compare = ttk.Button(action_frame, text="📊 RUN BENCHMARK COMPARISON", command=self.start_comparison)
        self.btn_compare.pack(ipadx=10, ipady=5, pady=5)

    def _generate_grid(self):
        # Clear old widgets
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
            
        n = self.n_var.get()
        m = self.m_var.get()
        
        self.entries_alloc = []
        self.entries_max = []
        self.entries_avail = []
        
        # Headers
        ttk.Label(self.matrix_frame, text="Allocation").grid(row=0, column=0, columnspan=m)
        ttk.Label(self.matrix_frame, text="Max Need").grid(row=0, column=m+1, columnspan=m)
        
        # Available Vector
        avail_frame = ttk.LabelFrame(self.matrix_frame, text="Available Resources")
        avail_frame.grid(row=n+2, column=0, columnspan=2*m+2, pady=10)
        
        for j in range(m):
            e = ttk.Entry(avail_frame, width=5)
            e.pack(side="left", padx=2)
            e.insert(0, "3") # Default
            self.entries_avail.append(e)

        # Grid Loop
        for i in range(n):
            row_alloc = []
            row_max = []
            
            # Allocation Entries
            for j in range(m):
                e = ttk.Entry(self.matrix_frame, width=4)
                e.grid(row=i+1, column=j, padx=1, pady=1)
                e.insert(0, "0") # Default 0
                row_alloc.append(e)
            
            # Spacer
            ttk.Label(self.matrix_frame, text="  |  ").grid(row=i+1, column=m)
            
            # Max Entries
            for j in range(m):
                e = ttk.Entry(self.matrix_frame, width=4)
                e.grid(row=i+1, column=m+1+j, padx=1, pady=1)
                e.insert(0, str(np.random.randint(1, 8))) # Random Defaults
                row_max.append(e)
                
            self.entries_alloc.append(row_alloc)
            self.entries_max.append(row_max)

    def get_data(self):
        try:
            n = self.n_var.get()
            m = self.m_var.get()
            alloc = [[int(e.get()) for e in row] for row in self.entries_alloc]
            mx = [[int(e.get()) for e in row] for row in self.entries_max]
            avail = [int(e.get()) for e in self.entries_avail]
            return n, m, alloc, mx, avail
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers only.")
            return None

    def start_procedure(self):
        data = self.get_data()
        if not data: return
        n, m, alloc, mx, avail = data
        
        # Disable UI
        self.btn_run.config(state="disabled")
        self.status_label.config(text="⚙️ TRAINING AI AGENT... (Please Wait)", foreground="blue")
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, ">>> Analyzing System State...\n")
        
        # Use Threading so GUI doesn't freeze
        t = threading.Thread(target=self.run_training_thread, args=(n, m, alloc, mx, avail))
        t.start()

    def run_training_thread(self, n, m, alloc, mx, avail):
        # 1. TRAIN
        self.q_table = train_agent_live(n, m, alloc, mx, avail)
        
        # 2. PREPARE ANIMATION
        self.current_alloc = np.array(alloc)
        self.current_max = np.array(mx)
        self.current_avail = np.array(avail)
        self.finished_mask = [False] * n
        
        # 3. TRIGGER ANIMATION ON MAIN THREAD
        self.after(0, self.start_animation_loop)

    def start_animation_loop(self):
        self.status_label.config(text="🚀 AUTO-PILOT ENGAGED", foreground="green")
        self.log_text.insert(tk.END, ">>> Training Complete. Executing optimal policy...\n\n")
        self.animate_step()

    def animate_step(self):
        if all(self.finished_mask):
            self.status_label.config(text="✅ ALL PROCESSES FINISHED", foreground="black")
            self.log_text.insert(tk.END, "\n>>> SYSTEM SAFE. EXECUTION COMPLETE.")
            self.btn_run.config(state="normal")
            return

        state = tuple(map(int, self.finished_mask))
        
        # AI Decision
        if state in self.q_table:
            action = np.argmax(self.q_table[state])
            
            # Visual feedback
            self.log_text.insert(tk.END, f"AI Agent chooses Process P{action}...\n")
            
            # Check safety (Just to be sure visually)
            n, m = self.n_var.get(), self.m_var.get()
            need = [self.current_max[action][j] - self.current_alloc[action][j] for j in range(m)]
            
            if all(need[j] <= self.current_avail[j] for j in range(m)):
                self.log_text.insert(tk.END, f"   -> Resources Granted. P{action} Finished.\n")
                self.log_text.insert(tk.END, f"   -> Resources Released: {self.current_alloc[action]}\n")
                
                # Update Data
                for j in range(m):
                    self.current_avail[j] += self.current_alloc[action][j]
                    
                self.finished_mask[action] = True
                self.log_text.insert(tk.END, f"   -> New Available: {self.current_avail}\n\n")
                self.log_text.see(tk.END)
                
                # Visual update delay
                self.after(1200, self.animate_step)
            else:
                self.log_text.insert(tk.END, f"❌ AI ERROR: P{action} creates Deadlock! (Unsafe State)\n")
                self.status_label.config(text="❌ DEADLOCK DETECTED", foreground="red")
                self.btn_run.config(state="normal")
        else:
             self.log_text.insert(tk.END, "❌ AI Confused (State not in training memory).\n")
             self.btn_run.config(state="normal")

    def start_comparison(self):
        """Runs the battle between Global and JIT agents"""
        # --- REPLACE YOUR OLD start_comparison WITH THESE 3 FUNCTIONS ---

    def start_comparison(self):
        """1. PREPARE AND START THE THREAD"""
        data = self.get_data()
        if not data: return
        
        # Disable button so user doesn't click twice
        self.btn_compare.config(state="disabled")
        
        # Show "Loading" message
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, ">>> INITIALIZING BENCHMARK ARENA...\n")
        self.log_text.insert(tk.END, ">>> Please wait. Loading/Training AI models...\n")
        self.update() # Force UI to show text immediately
        
        # Run the heavy work in a separate thread
        t = threading.Thread(target=self.run_benchmark_thread, args=(data,))
        t.start()

    def run_benchmark_thread(self, data):
        """2. DO THE HEAVY LIFTING (BACKGROUND THREAD)"""
        n, m, alloc, mx, avail = data
        
        # Initialize Arena (This might take 2-3 seconds if retraining)
        arena = BenchmarkArena(n, m)
        
        # Run Global Test
        g_results = arena.run_global_test(alloc, mx, avail)
        
        # Run JIT Test
        j_results = arena.run_jit_test(alloc, mx, avail)
        
        # When done, tell the Main Thread to show results
        self.after(0, lambda: self.finalize_benchmark(g_results, j_results))

    def finalize_benchmark(self, g_results, j_results):
        """3. UPDATE UI (MAIN THREAD)"""
        # Unpack the results
        g_success, g_time, g_steps, g_status = g_results
        j_success, j_time, j_steps, j_status = j_results
        
        # Re-enable the button
        self.btn_compare.config(state="normal")
        
        # Log completion
        self.log_text.insert(tk.END, ">>> BENCHMARK COMPLETE.\n")
        self.log_text.see(tk.END)
        
        # Trigger the popup window
        self.show_comparison_report(g_success, g_time, g_steps, g_status,
                                    j_success, j_time, j_steps, j_status)

    def show_comparison_report(self, g_success, g_time, g_steps, g_status,
                                     j_success, j_time, j_steps, j_status):
        
        # Formatting the results
        report_window = tk.Toplevel(self)
        report_window.title("Model Comparison Results")
        report_window.geometry("600x400")
        
        ttk.Label(report_window, text="Performance Benchmark", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create Treeview Table
        cols = ("Metric", "Global Agent (Offline)", "JIT Agent (Online)")
        tree = ttk.Treeview(report_window, columns=cols, show="headings", height=8)
        
        tree.heading("Metric", text="Metric")
        tree.heading("Global Agent (Offline)", text="Global Agent (Offline)")
        tree.heading("JIT Agent (Online)", text="JIT Agent (Online)")
        
        tree.column("Metric", width=150)
        tree.column("Global Agent (Offline)", width=200)
        tree.column("JIT Agent (Online)", width=200)
        
        # DATA ROWS
        
        # 1. Accuracy
        g_acc_icon = "❌ FAILED" if not g_success else "✅ PASSED"
        j_acc_icon = "❌ FAILED" if not j_success else "✅ PASSED"
        tree.insert("", "end", values=("Result", g_acc_icon, j_acc_icon))
        
        # 2. Time
        # Global is usually faster (no training), JIT slower (training overhead)
        tree.insert("", "end", values=("Total Time", f"{g_time:.4f} sec", f"{j_time:.4f} sec"))
        
        # 3. Status Detail
        tree.insert("", "end", values=("Exit Status", g_status, j_status))
        
        # 4. Path Length
        tree.insert("", "end", values=("Steps Taken", len(g_steps), len(j_steps)))
        
        # 5. The Path itself
        g_path_str = " -> ".join(g_steps) if len(g_steps) < 6 else " -> ".join(g_steps[:4]) + "..."
        j_path_str = " -> ".join(j_steps) if len(j_steps) < 6 else " -> ".join(j_steps[:4]) + "..."
        tree.insert("", "end", values=("Sequence", g_path_str, j_path_str))

        tree.pack(pady=20, padx=20, fill="both", expand=True)
        
        ttk.Button(report_window, text="Close", command=report_window.destroy).pack(pady=10)

if __name__ == "__main__":
    app = RLBankersApp()
    app.mainloop()