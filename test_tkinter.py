import tkinter as tk

def on_closing():
    print("Tkinter window closed")
    root.destroy()

print("Starting Tkinter test...")

root = tk.Tk()
root.title("Tkinter Test")
label = tk.Label(root, text="Tkinter is working!", font=("Arial", 14))
label.pack(pady=20)

root.protocol("WM_DELETE_WINDOW", on_closing)

print("Running Tkinter main loop...")
root.mainloop()
print("Tkinter test completed.")