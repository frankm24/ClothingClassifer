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
    tiles = []

    def left_button_down(self, event=None):
        self.left_button = "down"
        print("Left button down fired.")
        
    def left_button_up(self, event=None):
        self.left_button = "up"
        print("Left button up fired.")

    def motion(self, event=None):
        if self.left_button == "down": 
            tile = event.widget.winfo_containing(event.x_root, event.y_root)
            print(tile)
            tk.itemconfig(tile, bg = "white")
             
    def clickExitButton(self):
        exit()
        
    def __init__(self, root):
        for r in range(28):
            for c in range(28):
                 tile = Frame(root, bd = '5', bg = "black", width = 20, height = 20
                   )
                 tile.grid(row=r,column=c)
                 self.tiles.insert(len(self.tiles), tile)
        for tile in self.tiles:
            print(tile)
            #tile.bind("<Enter>", lambda event, arg=tile: self.onMbEnter(arg, event))
            
        root.bind("<ButtonPress-1>", self.left_button_down)
        root.bind("<ButtonRelease-1>", self.left_button_up)
        root.bind("<Motion>", self.motion)
        
root = tk.Tk()
app = Draw(root)

    #use "<Enter>" and "<Leave>" events for detecting mouse
class Paint: 
    drawing_tool = "pencil" #Pencil or eraser for my implementation.
    left_button = "up"
    x_pos, y_pos = None, None

    def left_button_down(self, event=None):
        self.left_button = "down"
        
    def left_button_up(self, event=None):
        self.left_button = "up"
        self.x_pos = None
        self.y_pos = None
        
    def motion(self, event=None):
        self.pencil_draw(event)

    def pencil_draw(self, event=None):
        if self.left_button == "down":
            if self.drawing_tool == "pencil":
                if self.x_pos is not None and self.y_pos is not None:
                    event.widget.create_line(self.x_pos, self.y_pos, event.x, event.y,
                        smooth=True, fill="white"
                    )
            self.x_pos = event.x
            self.y_pos = event.y
            
    def __init__(self, root):
        drawing_area = Canvas(root, bg = "black", bd=0, width=28, height=28)
        drawing_area.pack()
        drawing_area.bind("<Motion>", self.motion)
        drawing_area.bind("<ButtonPress-1>", self.left_button_down)
        drawing_area.bind("<ButtonRelease-1>", self.left_button_up)       
        drawing_area.scale("all", 0 , 0, 10, 10)
#root = Tk()
#paint = Paint(root)

#Load training data using library that interprets MNIST data files automatically.
#Otherwise, I would have to write code that interprets a custom file type, which
#I have no idea how to do.

data = MNIST("TrainingData")
images, labels = data.load_training()

#GUI buttons
layout = [
    [sg.Text("Hand Written Digit Guessing NN",
             font = (None, 30),
             pad = (0,0),
             background_color = "Black",
             justification = "center")],
    [sg.Button("Close")],
    [sg.Button("Enter")],         
]

window = sg.Window("Hand Written Digit Guesser",
    layout,
    margins = (200, 200),
    background_color = "Black",
)
print(dir(sg))
#for case in zip(labels, images):
#    print(case)
    
#Loop which makes the PySimpleGUI interface work by reading user interactions  
while True:
    event, values = window.read() #read() returns info about next interaction, when it occurs
    if event == "Close" or event == sg.WIN_CLOSED:
        break
window.close()
 
