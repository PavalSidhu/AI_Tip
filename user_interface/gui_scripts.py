import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import messagebox
import sys

sys.path.insert(0, '/home/pavalsidhu/AI_TIP/evaluate_log')
from evaluate_log import predict_log_file

# Define the severity mapping globally
severity_mapping = {
    "Analysis": "Low",
    "Backdoor": "Critical",
    "Backdoors": "Critical",
    "DoS": "Medium",
    "Exploits": "High",
    "Fuzzers": "Medium",
    "Generic": "Informational",
    "Reconnaissance": "Low",
    "Shellcode": "Critical",
    "Worms": "Critical",
}

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Log File Predictor")
        # Make the application window smaller
        self.geometry("800x500")

        self.choose_file_btn = tk.Button(self, text="Choose Log File", command=self.choose_file)
        self.choose_file_btn.pack(pady=10)

        self.display = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=100, height=25,
                                                 borderwidth=2, relief="groove", padx=5, pady=5,
                                                 font=("Consolas", 10), spacing3=5)
        self.display.pack(padx=10, pady=10)

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                predictions = predict_log_file(file_path, '/home/pavalsidhu/AI_TIP/ai_model/models/attack_cat_RandomForestClassifier_model_nb_final.pkl')
                self.display_predictions(predictions)
            except Exception as e:
                messagebox.showerror("Error", "Failed to process file.\n" + str(e))

    def display_predictions(self, predictions):
        self.display.delete('1.0', tk.END)  # Clear the display before adding new content
        for i, prediction in enumerate(predictions):
            # Construct each row's text with additional padding lines to simulate a top and bottom border
            display_text = f"\nRow {i}: {prediction}\n\n"
            self.display.insert(tk.END, display_text, 'tag-border')

        # Configure a tag to adjust the background color for visual separation
        self.display.tag_configure('tag-border', background="#e0e0e0", spacing3=2)
