import tkinter as tk
from tkinter import filedialog, Label, font
from PIL import Image, ImageTk
from main import run, is_model_trained, train_model
from config import DEFAULT_IMAGE_FILE

def main():

    def upload_image():
        file_path = filedialog.askopenfilename()
        try:
            prediction = run(file_path, False)
            display_image(Image.open(file_path))
        except ValueError as e:
            prediction = str(e)

        show_prediction(prediction)

    def show_prediction(prediction):
        status_label.config(text=prediction)

    def display_image(img):
        display_img = img.resize((250, 250), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(display_img)
        
        # Save a reference to the image to avoid it being deleted by the garbage collector
        label_image.image = img_tk  
        label_image.config(image=img_tk)
        
    app = tk.Tk()
    app.title("CXR Disease Detection")
    app.configure(bg='#212121') 

    f = font.Font(family='Assistant', size=14)
    label=Label(app, text="Please upload a chest x-ray scan \n for detection", bg='#212121', fg="white", font=f).pack()

    # Label to display the image
    label_image = tk.Label(app)

    bg_img = Image.open(DEFAULT_IMAGE_FILE)
    display_img = bg_img.resize((250, 250), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(display_img)
    label_image.config(image=img_tk, borderwidth=0, highlightthickness=0)
     # Maintain reference because otherwise garbage collector deletes it and the image doesnt show up:(
    label_image.image = img_tk 
    label_image.pack(pady=5)

    # Button to upload an image
    button_upload = tk.Button(app, text="Upload Image", font=f, bg='#a9e5e5', command=upload_image)
    button_upload.pack()

    # Prediction label
    status_label = tk.Label(app, text="", fg="red", font=f, background='#212121')
    status_label.pack(pady=10)

    return app

if __name__ == "__main__":
    # If the model hasn't been trained yet, GUI will not work. First needs to train and only after the GUI will pop up.
    if not is_model_trained():
        train_model()

    # Start GUI after training is confirmed
    gui_app = main()
    gui_app.mainloop()