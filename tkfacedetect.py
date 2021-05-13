from tkinter import *
from tkinter.ttk import *
import tkinter as tk 
from tkinter import Message, Text 
from tkinter import filedialog
import tkinter.ttk as ttk 
import tkinter.font as font 
from PIL import Image,ImageTk
import itertools  
import face_recognition
import cv2
import os

TOLERANCE = 0.542312 #niższa tolerancja jest "surowsza", uzywana do sprawdzania jak bardzo dwa encodingi twarzy się różnią od siebie (dystans)
FRAME_THICKNESS = 3
FONT_THICKNESS = 1
MODEL = "hog"  #histogram of oriented gradients, moze też być np cnn czyli  Convolutional Neural Network, ale żeby poprawnie używać trzeba mieć biblioteke dlib dobrze skonfig

known_faces = []
known_names = []

window = tk.Tk()
window.resizable(width=True, height=True)
window.configure(width=1550, height=750)
window.title("SRT") 
window.configure(background ='black') 
window.grid_rowconfigure(0, weight = 1) 
window.grid_columnconfigure(0, weight = 1) 

message = tk.Label( 
    window, text ="SYSTEM ROZPOZNAWANIA TWARZY",  
    bg ="green", fg = "white", width = 50,  
    height = 3, font = ('consolas', 30, 'bold'))  
      
message.place(x = 200, y = 20) 

def load_known_faces_dir():
    global KNOWN_FACES_DIR
    KNOWN_FACES_DIR = filedialog.askdirectory()
    
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
            image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)

def load_unknown_faces_dir():
    global UNKNOWN_FACES_DIR
    UNKNOWN_FACES_DIR = filedialog.askdirectory()

    for filename in os.listdir(UNKNOWN_FACES_DIR):
        print(filename)
        image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
        locations = face_recognition.face_locations(image, model=MODEL)                 #znajdowanie twarzy
        encodings = face_recognition.face_encodings(image, locations)                   #konwersja na 128bitowy format uzywany do trenowania?
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                  #konwersja z RGB na BGR
                                                                                        #The BGR is a 24-bit representation where the lower-addressed 8 bits are blue, the next-addressed 8 are green and higher-addressed 8 are red.
    
        for face_encoding, face_location in zip(encodings,locations):
            results = face_recognition.compare_faces(known_faces,face_encoding,TOLERANCE)       #porównujemy jeden encoding znanej twarzy z jednym encodingiem nieznanej twarzy
            match = None
            if True in results:
                match = known_names[results.index(True)]
                print(f"Match found : {match}")

                top_left = (face_location[3], face_location[0])         #koordynaty do obrysu twarzy
                bottom_right = (face_location[1], face_location[2])

                color = [255,255,255]
                cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
                
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2]+40)
            
                cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)                                                                 #
                cv2.putText(image,match,(face_location[3], face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),FONT_THICKNESS)       #wstawianie opisu

        cv2.imshow(filename,image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)

def load_images():
    global images
    images = filedialog.askopenfilenames(filetypes=[("cokolwiek","*.jpg")])
    
    if len(images) > 1:              
        for image in images:
            print(image)

def show_images():

    load_images()

    ob=images
    iterator = iter(ob)
    iterator = itertools.cycle(iterator)

    new_window = Toplevel(window)   #tworzymy nowe okno które będzie zawsze ponad oknem window
    new_window.geometry('1600x900') 
    
    panel = tk.Label(new_window)
    panel.pack()

    def next_img():
        try:
            img = next(iterator) 
        except StopIteration:
            return   
        
        img = Image.open(img)               #wyswietlanie obrazu w nowo otwartym oknie
        img = ImageTk.PhotoImage(img)
        panel.img = img  
        panel['image'] = img


    btn = tk.Button(new_window,text='Następny obraz', command=next_img,
    anchor=tk.S,fg ="white", bg ="green",  
    width = 20, height = 3, activebackground = "Red",  
    font =('consolas', 15, ' bold '))
    btn.pack(fill='none',side="bottom")

    next_img()

    new_window.mainloop()



train_face_button = tk.Button(window, text ="Trenuj znane twarze",  
command = load_known_faces_dir, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('consolas', 15, ' bold ')) 
train_face_button.place(x = 350, y = 500)

find_face_button = tk.Button(window, text ="Znajdź nieznane twarze",  
command = load_unknown_faces_dir, fg ="white", bg ="green",  
width = 22, height = 3, activebackground = "Red",  
font =('consolas', 15, ' bold ')) 
find_face_button.place(x = 615, y = 500) 

quit_window = tk.Button(window, text ="Wyjdź",                    
command = window.destroy, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('consolas', 15, ' bold ')) 
quit_window.place(x = 900, y = 500) 
  
show_img = tk.Button(window, text ="Wyświetl obrazy",           #okienko wyświetlania obrazów
command = show_images, fg ="white", bg ="green",  
width = 20, height = 3, activebackground = "Red",  
font =('consolas', 15, ' bold ')) 
show_img.place(x = 625, y = 600) 



window.mainloop() 