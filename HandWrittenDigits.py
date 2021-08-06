import PySimpleGUI as sg
import numpy as np
import random
from mnist import MNIST
import tkinter as tk

'''
PySimpleGUI -- old library, stopped using because it is too abstract/not enough features to make the
project.
Tkinter -- good basic UI designing library.
Numpy -- has np arrays, matrix multply functions, and sother tuff I may want
Random -- used to generate random numbers
MNIST -- Pure python library which interprets traning data weird file formats from the MNIST database

Using grid of tkinter Frames instead of Canvas for drawing UI because I can make the individual frames
look like pixels and make each "pixel" bigger than a display pixel. A 28x28 canvas can only be 28x28
display pixels in size. With a grid of frames, I can easily change the displayed size of each "pixel"
while maintaining thw 28x28 drawing resolution and make it easier for people to draw hand written digts
on the 28x28 grid format that the network will accept.

https://stackoverflow.com/questions/51369844/how-to-trigger-tkinters-enter-event-with-mouse-down
Above link is why I had to write it this way.
'''

class Draw:
    left_button = "up"
    draw_strength = 1
    draw_channel = 255 * draw_strength 
    tiles = []
    x_pos = None
    y_pos = None

    def left_button_down(self, event=None):
        self.left_button = "down"
        if self.left_button == "down":
            tile = event.widget.winfo_containing(event.x_root, event.y_root)
            if type(tile) == tk.Frame and tile in self.tiles:
                #index = tiles.index(tile)
                tk.Frame.configure(tile, bg = "#ffffff")  #ededed
                
        
    def left_button_up(self, event=None):
        self.left_button = "up"

    def b1motion(self, event=None):
        tile = event.widget.winfo_containing(event.x_root, event.y_root)
        if type(tile) == tk.Frame and tile in self.tiles:
            tk.Frame.configure(tile, bg = "#ffffff")
            
    def clearDrawing(self, event=None):
        for tile in self.tiles:
            tk.Frame.configure(tile, bg = "black")
    
    def clickExitButton(self):
        exit()
        
    def __init__(self, root):
        #Titles and buttons
        top_frame = tk.Frame(root, bg = "black")
        drawing_frame = tk.Frame(root, bg = "black", cursor = "pencil", highlightbackground = "white", highlightthickness = 2)
        bottom_frame = tk.Frame(root, bg = "black")
        
        title = tk.Label(top_frame, text = "Hand Written Digits Nueral Network", fg = "white",
                    bg = "black", font = (None, 30))
        info = tk.Label(top_frame, text = "Draw a digit and the nueral network will guess what it is!",
                        fg = "white", bg = "black")
        
        info.pack(side = "bottom")
        title.pack(side = "top")
        
        clear_button = tk.Frame(bottom_frame, bg = "black", cursor = "hand2")
        clear_button_label = tk.Label(clear_button, text = "Clear", bg = "white")
        clear_button.pack(pady = 5)
        clear_button_label.pack(side = "bottom")
        
        #Make grid of tiles
        for r in range(28):
            for c in range(28):
                 tile = tk.Frame(drawing_frame, bd = '5', bg = "black", width = 20, height = 20)
                 tile.grid(row=r,column=c)
                 self.tiles.insert(len(self.tiles), tile)
            #tile.bind("<Enter>", lambda event, arg=tile: self.onMbEnter(arg, event))
            
        root.bind("<ButtonPress-1>", self.left_button_down)
        root.bind("<ButtonRelease-1>", self.left_button_up)
        root.bind("<B1-Motion>", self.b1motion)
        clear_button_label.bind("<ButtonPress-1>", self.clearDrawing)
        
        top_frame.pack(side = "top", fill = "x")
        drawing_frame.pack()
        bottom_frame.pack(side = "bottom")
        
root = tk.Tk()
root.title("Hand Written Digit NN")
root.configure(bg = "black")

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
#    print(case)
    

