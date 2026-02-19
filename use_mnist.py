from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageFilter

model = load_model('mnist_model.keras')
canvas_size = 280
image_size = 28
brush_size = 8

root = tk.Tk()
root.title("MNIST Digit Recognizer")

canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack()

image = Image.new("L", (canvas_size, canvas_size), 0)
draw = ImageDraw.Draw(image)

def draw_digit(event):
    x, y = event.x, event.y
    r = brush_size
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

canvas.bind("<B1-Motion>", draw_digit)

def clear():
    canvas.delete("all")
    global image, draw
    image = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(image)

def predict():
    img = image.copy()

    # 1. Get bounding box of the digit
    bbox = img.getbbox()
    if not bbox:
        return

    img = img.crop(bbox)

    # 2. Resize while keeping aspect ratio
    w, h = img.size
    if w > h:
        new_w = 20
        new_h = int(20 * h / w)
    else:
        new_h = 20
        new_w = int(20 * w / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 3. Paste into 28x28 image (centered)
    new_img = Image.new("L", (28, 28), 0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    new_img.paste(img, (x_offset, y_offset))

    # 4. Blur slightly (MNIST softness)
    new_img = new_img.filter(ImageFilter.GaussianBlur(1))

    # 5. Normalize
    img_arr = np.array(new_img) / 255.0
    img_arr = img_arr.reshape(1, 784)

    # Debug view (keep this, it's gold)
    plt.imshow(img_arr.reshape(28, 28), cmap="gray")
    plt.title("Model Input")
    plt.axis("off")
    plt.show()

    prediction = model.predict(img_arr)
    print("Predicted digit:", np.argmax(prediction))

    

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Predict", command=predict).pack(side="left")
tk.Button(btn_frame, text="Clear", command=clear).pack(side="left")

root.mainloop()

#==============================#

# img = image.load_img("num_test.png", color_mode="grayscale", target_size=(28, 28))
# img = image.img_to_array(img)
# img = img / 255.0
# img = img.reshape(1, 784)

# print("Predicted digit:", model.predict(img).argmax())
