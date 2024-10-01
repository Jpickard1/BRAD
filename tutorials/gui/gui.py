import os
import sys
sys.path.append('../../RAG-DEV/')

import tkinter as tk
from tkinter import scrolledtext, ttk
from BRAD import brad
from BRAD import llms

class ChatBotGUI:
    def __init__(self, master, bot):
        self.master = master
        master.title("BRAD Chatbot")

        # Dark grey background for the entire window
        master.configure(bg='#2E2E2E')

        # Initialize chatbot instance
        self.bot = bot

        # Configure grid to make resizing proportional
        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)
        
        # Arial font
        self.font_style = ('Arial', 12)

        # Chat display area (scrollable text)
        self.chat_display = scrolledtext.ScrolledText(master, wrap=tk.WORD, state='disabled', height=20, width=80, bg='#3C3C3C', fg='white', font=self.font_style)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # User input area
        self.user_input = tk.Entry(master, width=70, font=self.font_style, bg='#4F4F4F', fg='white')
        self.user_input.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.user_input.bind("<Return>", self.send_message)  # Bind Enter key to send message

        # Send button
        self.send_button = tk.Button(master, text="Send", command=self.send_message, font=self.font_style, bg='#6E6E6E', fg='white')
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Make the input box and button resize proportionally
        master.grid_columnconfigure(0, weight=1)

        # Add toggle button for expanding/collapsing configuration panel
        self.toggle_config_button = tk.Button(master, text="Config", command=self.toggle_config_panel, font=self.font_style, bg='#6E6E6E', fg='white')
        self.toggle_config_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Create a frame for the right-hand side config panel, initially hidden
        self.config_frame = ttk.Frame(master)
        self.config_panel_visible = False

        # Create configuration panel content
        self._create_config_panel()

    def _create_config_panel(self):
        """Create UI elements for modifying bot configuration."""
        config = self.bot.chatstatus['config']

        # Debug mode (Boolean checkbox) with smaller font
        self.debug_var = tk.BooleanVar(value=config.get('debug', False))
        self.debug_checkbox = ttk.Checkbutton(self.config_frame, text='Debug Mode', variable=self.debug_var)
        self.debug_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Python path (Entry for modifying path) with smaller font
        ttk.Label(self.config_frame, text="Python Path:", font=('Arial', 10)).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.py_path_var = tk.StringVar(value=config.get('py-path', ''))
        self.py_path_entry = ttk.Entry(self.config_frame, textvariable=self.py_path_var, width=30, font=('Arial', 10))
        self.py_path_entry.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        # Apply button with smaller font
        self.apply_button = ttk.Button(self.config_frame, text="Apply Changes", command=self._apply_config_changes)
        self.apply_button.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

    def toggle_config_panel(self):
        """Toggle the visibility of the configuration panel."""
        if self.config_panel_visible:
            # If the panel is visible, hide it
            self.config_frame.grid_forget()
        else:
            # Show the panel if hidden
            self.config_frame.grid(row=0, column=2, rowspan=3, padx=10, pady=10, sticky="nsew")
        self.config_panel_visible = not self.config_panel_visible

    def _apply_config_changes(self):
        """Apply changes to the bot's config dictionary."""
        self.bot.chatstatus['config']['debug'] = self.debug_var.get()
        self.bot.chatstatus['config']['py-path'] = self.py_path_var.get()

    def send_message(self, event=None):
        """Send user message to the chatbot and display the response."""
        user_message = self.user_input.get().strip()
        if user_message:
            # Display user message in chat window
            self._insert_message(f'You: {user_message}\n', align='right', bg_color='#D3D3D3', fg_color='black')

            # Clear input field
            self.user_input.delete(0, tk.END)

            # Disable send button and change its color while waiting for the bot response
            self.send_button.config(state='disabled', bg='lightgrey')

            # Simulate chatbot processing time
            self.master.after(500, self.process_response, user_message)

    def process_response(self, user_message):
        """Process the chatbot's response after the delay."""
        # Send message to chatbot and get response
        response = self.bot.invoke(user_message)
        if not response:
            response_text = "Goodbye!"
        else:
            response_text = self.bot.chatstatus['output']

        # Display chatbot's response
        self._insert_message(f'BRAD: {response_text}\n', align='left', bg_color='#3C3C3C', fg_color='white')

        # Re-enable send button and restore its color
        self.send_button.config(state='normal', bg='#6E6E6E')

    def _insert_message(self, message, align='left', bg_color='#FFFFFF', fg_color='black'):
        """Inserts a message into the chat display area with specified alignment and color."""
        self.chat_display.config(state='normal')  # Enable editing of chat display

        # Add colored background and text color for the message
        self.chat_display.tag_configure(align, justify=align, background=bg_color, foreground=fg_color)
        self.chat_display.insert(tk.END, message, align)
        
        # Auto-scroll to the bottom
        self.chat_display.yview(tk.END)
        self.chat_display.config(state='disabled')  # Disable editing of chat display

# Create the chatbot instance
# Initialize a chatbot
llm = llms.load_nvidia()
bot = brad.chatbot(llm=llm)

# Create main window
root = tk.Tk()
root.geometry("800x500")  # Set a default window size
chatbot_gui = ChatBotGUI(root, bot=bot)

# Start the GUI event loop
root.mainloop()

