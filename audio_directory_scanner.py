import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from mutagen import File as MutagenFile
import csv
from datetime import datetime

class DirectoryScannerApp:
    def __init__(self, root):
        self.root = root
        self.progress_bar = None

    def run(self):
        self.root.withdraw()  # Hide the root window
        self.root.after(0, self.start_scan)  # Start the scanning process

    def start_scan(self):
        try:
            directory = self.select_directory()
            if not directory:
                self.show_warning("No Directory Selected", "You must select a directory to scan.")
                return

            file_data, max_depth = self.scan_directory(directory)

            if not file_data:
                self.show_warning("No Files Found", "No files were found in the selected directory.")
                return

            self.save_to_csv(file_data, max_depth, directory)

        except Exception as e:
            self.show_error("Error", f"An unexpected error occurred:\n{str(e)}")
        finally:
            self.exit_program()

    def select_directory(self):
        return filedialog.askdirectory(title="Select Directory to Scan")

    def scan_directory(self, directory):
        file_data = []
        max_depth = 0

        def scan_subdirectory(parent_chain, sub_dir, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)

            try:
                for idx, item in enumerate(os.listdir(sub_dir)):
                    item_path = os.path.join(sub_dir, item)
                    if os.path.isdir(item_path):
                        scan_subdirectory(parent_chain + [(os.path.basename(sub_dir), "folder")], item_path, depth + 1)
                    else:
                        item_name = os.path.basename(item)
                        item_type = Path(item).suffix if Path(item).suffix else "file"
                        duration = self.get_file_duration(item_path) if item_type.lower() in ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.mp4', '.aac', '.aiff', '.opus'] else ""
                        file_data.append(parent_chain + [(item_name, item_type), duration])

                    # Update progress bar and process GUI events
                    if idx % 100 == 0:  # Update less frequently
                        self.update_progress_bar()
                        self.root.update()  # Keep the GUI responsive

            except Exception as e:
                self.show_error("Error", f"An error occurred while scanning {sub_dir}:\n{str(e)}")

        # Initialize progress bar
        self.init_progress_bar(len(list(os.walk(directory))))
        scan_subdirectory([(os.path.basename(directory), "folder")], directory, 1)
        self.destroy_progress_bar()
        return file_data, max_depth

    def get_file_duration(self, file_path):
        try:
            audio = MutagenFile(file_path)
            if audio and audio.info and hasattr(audio.info, 'length'):
                duration_seconds = audio.info.length  # Duration in seconds
                return self.format_duration(duration_seconds)
            return ""
        except Exception as e:
            return ""  # Gracefully skip and return an empty string for files that can't be processed

    @staticmethod
    def format_duration(seconds):
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02}:{remaining_seconds:02}"

    def save_to_csv(self, file_data, max_depth, directory):
        try:
            # Find the maximum number of levels
            max_levels = max(len(row) - 1 for row in file_data)  # Subtract 1 to ignore the duration column
            headers = []
            for i in range(max_levels):
                headers.extend([f"Level_{i+1}", f"Level_{i+1}_type"])
            headers.append("Duration")

            # Generate CSV file name based on the directory name and current date/time
            base_dir_name = os.path.basename(directory)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(directory, f"{base_dir_name}_{current_time}.csv")

            with open(output_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)  # Write headers once

                # Append data to CSV
                for row in file_data:
                    # Flatten the row, ensuring each level and its type are correctly placed
                    flat_row = []
                    for item in row[:-1]:  # All elements except the duration
                        flat_row.extend(item)
                    # Ensure the row has the correct number of columns by adding empty cells if needed
                    while len(flat_row) < len(headers) - 1:  # Subtract 1 to leave space for duration
                        flat_row.append("")
                    flat_row.append(row[-1])  # Add the duration at the end
                    writer.writerow(flat_row)

            self.show_info("Success", f"CSV file saved successfully to {output_file}")

        except Exception as e:
            self.show_error("Error", f"An error occurred while saving the CSV file:\n{str(e)}")

    def init_progress_bar(self, max_value):
        self.progress_bar = tk.Toplevel(self.root)
        self.progress_bar.title("Scanning...")
        self.progress_bar.geometry("300x50")
        progress = ttk.Progressbar(self.progress_bar, orient="horizontal", length=280, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = max_value
        progress["value"] = 0
        self.progress = progress
        self.progress_bar.update()

    def update_progress_bar(self):
        if self.progress_bar:
            self.progress["value"] += 1
            self.progress_bar.update()

    def destroy_progress_bar(self):
        if self.progress_bar:
            self.progress_bar.destroy()
            self.progress_bar = None

    def exit_program(self):
        self.root.quit()  # Close the Tkinter main loop
        self.root.destroy()  # Destroy the main window
        print("Exiting gracefully.")  # Optional: Log to console for confirmation

    @staticmethod
    def show_info(title, message):
        messagebox.showinfo(title, message)

    @staticmethod
    def show_warning(title, message):
        messagebox.showwarning(title, message)

    @staticmethod
    def show_error(title, message):
        messagebox.showerror(title, message)


def main():
    root = tk.Tk()
    app = DirectoryScannerApp(root)
    app.run()
    root.mainloop()  # Start the Tkinter event loop


if __name__ == "__main__":
    main()
