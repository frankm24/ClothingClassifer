import PySimpleGUI as sg
import numpy as np
import random
from mnist import MNIST
import tkinter as tk

'''
Libraries used:

PySimpleGUI -- old library, stopped using because it is too abstract/not enough features to make the
project.
Tkinter -- basic UI designing library.
Numpy -- has np arrays, matrix multply functions, and other stuff I may want for math
Random -- used to generate random numbers
MNIST -- Pure python library which interprets the weird file formats of the training data from the MNIST database

Using grid of tkinter Frames instead of Canvas for drawing UI because I can make the individual frames
look like pixels and make each "pixel" bigger than a display pixel. A 28x28 canvas can only be 28x28
display pixels in size. With a grid of frames, I can easily change the displayed size of each "pixel"
while maintaining thw 28x28 drawing resolution and make it easier for people to draw hand written digts
on the 28x28 grid format that the network will accept.

This code is not perfect but it works.
'''

class Draw:
    left_button = "up"
    status = "idle"
    draw_color = "#ffffff"
    draw_color2 = "#ffffff"
    enter_button_label = None
    input_drawing = []
    tiles = [[0 for c in range(28)] for r in range(28)] #https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/
    output_labels = []
    x_pos = None
    y_pos = None

    def left_button_down(self, event=None):
        self.left_button = "down"
        if self.left_button == "down":
            tile = event.widget.winfo_containing(event.x_root, event.y_root)
            info = tile.grid_info()
            if info and type(tile) == tk.Frame:
                tk.Frame.configure(tile, bg = self.draw_color)  
                r = info["row"]
                c = info["column"]
                #color touching tiles
                right = self.tiles[r][c+1]
                above = self.tiles[r+1][c]
                left = self.tiles[r][c-1]
                below = self.tiles[r-1][c]
                tk.Frame.configure(right, bg = self.draw_color2)
                tk.Frame.configure(above, bg = self.draw_color2)
                tk.Frame.configure(left, bg = self.draw_color2)
                tk.Frame.configure(below, bg = self.draw_color2)   
                
    def left_button_up(self, event=None):
        self.left_button = "up"

    def b1motion(self, event=None):
            tile = event.widget.winfo_containing(event.x_root, event.y_root)
            info = tile.grid_info()
            if info and type(tile) == tk.Frame:
                tk.Frame.configure(tile, bg = self.draw_color)  
                r = info["row"]
                c = info["column"]
                #color touching tiles
                right = self.tiles[r][c+1]
                above = self.tiles[r+1][c]
                left = self.tiles[r][c-1]
                below = self.tiles[r-1][c]
                tk.Frame.configure(right, bg = self.draw_color2)
                tk.Frame.configure(above, bg = self.draw_color2)
                tk.Frame.configure(left, bg = self.draw_color2)
                tk.Frame.configure(below, bg = self.draw_color2) 
            
    def clearDrawing(self, event=None):
        for r in range(28):
            for c in range(28):
                tile = self.tiles[r][c]
                tk.Frame.configure(tile, bg = "#000000")
            
    def onEnter(self, event=None):
        self.enter_button_label.configure(text = "Guessing...")
        for r in range(28):
            for c in range(28):
                tile = self.tiles[r][c]
                #Convert hex to brightness value used by the network
                #https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
                brightness = int(tile["bg"].lstrip('#')[:2], 16)
                self.input_drawing.insert(len(self.input_drawing), brightness)
        print(self.input_drawing)
        #Make sure the data is not messed up somehow
        assert(len(self.input_drawing) == 784)

        #feed forward
        
        #Reset
        self.input_drawing.clear()
        self.enter_button_label.configure(text = "Enter")

                
                
                            
        
    def __init__(self, root):
        #Title and top text
        top_frame = tk.Frame(root, bg = "#000000")
        drawing_frame = tk.Frame(root, bg = "#000000", cursor = "pencil", highlightbackground = "#ffffff",
                                 highlightthickness = 2)
        bottom_frame = tk.Frame(root, bg = "#000000")
        right_frame = tk.Frame(root, bg = "#000000")
        
        title = tk.Label(top_frame, text = "Hand Written Digits Nueral Network", fg = "#ffffff",
                         bg = "#000000", font = (None, 30))
        info = tk.Label(top_frame, text = "Draw a digit and the nueral network will guess what it is! (Train it first)",
                        fg = "#ffffff", bg = "#000000")  
        info.pack(side = "bottom")
        title.pack(side = "top")
        #Buttons
        clear_button = tk.Frame(bottom_frame, bg = "#000000", cursor = "hand2")
        clear_button_label = tk.Label(clear_button, text = "Clear", bg = "#ffffff")
        clear_button_label.pack()
        
        enter_button = tk.Frame(bottom_frame, bg = "#000000", cursor = "hand2")
        self.enter_button_label = tk.Label(enter_button, text = "Enter", bg = "#ffffff")
        self.enter_button_label.pack()
        
        clear_button.grid(row = 0, column = 0, padx = 5, pady = 5)
        enter_button.grid(row = 0, column = 1, padx = 5, pady = 5)

        
        #Make grid of drawing tiles
        for r in range(28):
            for c in range(28):
                 tile = tk.Frame(drawing_frame, bd = '5', bg = "#000000", width = 20, height = 20)
                 tile.grid(row=r,column=c)
                 self.tiles[r][c] = tile
        
        #Grid list of labels for probabilities/output of nueral net
        output_title = tk.Label(right_frame, text = "Output:", bg = "#000000", fg = "#ffffff")
        for r in range(10):
            label = tk.Label(right_frame, text = str(r) + ": 0%", bg = "#000000", fg = "#ffffff")
            label.grid(row = r, column = 0)
            self.output_labels.insert(len(self.output_labels), label)

        #Button for training 
        train_button = tk.Frame(right_frame, bg = "#000000", cursor = "hand2", pady = 5, padx = 10)
        train_button_label = tk.Label(train_button, text = "Start Training (MNIST)", bg = "#ffffff")
        train_button_label.pack()
        train_button.grid(row = 10, column = 0)

        #Button for saving data
        save_button = tk.Frame(right_frame, bg = "#000000", cursor = "hand2", pady = 5, padx = 10)
        save_button_label = tk.Label(save_button, text = "Save Network", bg = "#ffffff")
        save_button_label.pack()
        save_button.grid(row = 11, column = 0)

        #Button for loading data
        load_button = tk.Frame(right_frame, bg = "#000000", cursor = "hand2", pady = 5, padx = 10)
        load_button_label = tk.Label(load_button, text = "Load Network", bg = "#ffffff")
        load_button_label.pack()
        load_button.grid(row = 12, column = 0)

        #Button for saying the network guessed wrong
        incorrect_button = tk.Frame(right_frame, bg = "#000000", cursor = "hand2", pady = 5, padx = 10)
        incorrect_button_label = tk.Label(incorrect_button, text = "Guess Incorrect?", bg = "#ffffff")
        incorrect_button_label.pack()
        incorrect_button.grid(row = 13, column = 0)
        
        #event bindings
        root.bind("<ButtonPress-1>", self.left_button_down)
        root.bind("<ButtonRelease-1>", self.left_button_up)
        root.bind("<B1-Motion>", self.b1motion)
        clear_button_label.bind("<ButtonPress-1>", self.clearDrawing)
        self.enter_button_label.bind("<ButtonPress-1>", self.onEnter)
        #organize sections of GUI
        top_frame.pack(side = "top", fill = "x")
        right_frame.pack(side = "right", padx = 5)
        bottom_frame.pack(side = "bottom")
        drawing_frame.pack(padx = 5)
        print(drawing_frame.grid_info())
    
root = tk.Tk()
root.title("Hand Written Digit NN")
root.configure(bg = "#000000")
    
data = MNIST("TrainingData")
images, labels = data.load_training()

app = Draw(root)
tk.mainloop()

'''
#Load training data using library that interprets MNIST data files automatically.
#Otherwise, I would have to write code that interprets a custom file type, which
#I have no idea how to do.
'''



#for case in zip(labels, images):
#   print(case)
#   label = case[0]
#   image = case[1]


    

