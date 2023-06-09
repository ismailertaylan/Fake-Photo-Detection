# lib
import os
import numpy as np
import urllib

from matplotlib.figure import Figure

import tkinter as tk
import seaborn as sns
from urllib.request import urlopen
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image, ImageChops, ImageEnhance
from PIL.ExifTags import TAGS
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# global image
min_size = 5000  # Min size
max_size = 5000000  # Max size
background_image = "tkinter/background.png"
insert_image = "tkinter/insertimage.png"


class MetadataAnalysis:
    # EXIF BİLGİLERİNİ BASTIRMA
    def get_exif(image_path):
        with Image.open(image_path) as img:
            metadata = img._getexif()
            if metadata:
                exif_data = {}
                for tag_id, value in metadata.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
                return exif_data
            else:
                print("Resimde EXIF verisi bulunamadı.")
                return "Resimde EXIF verisi bulunamadı."

    # finding metadata of the image
    def analyze_metadata(image_path):
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id in exif_data:
                        tag = TAGS.get(tag_id, tag_id)
                        data = exif_data.get(tag_id)
                        if tag == "Software":
                            print("The image contains software metadata and may be edited.")
                            return "The image contains software metadata and may be edited."

                    print("The image does not contain any known software metadata and may be authentic.")
                    return "The image does not contain any known software metadata and may be authentic."

                else:
                    print("The image does not contain any metadata.")
                    return "The image does not contain any metadata."

        except Exception as e:
            print("Failed to analyze metadata: ", e)
            return "Failed to analyze metadata"


class ELAAnalysis:
    def convert_to_ela_image(path, quality):
        temp_filename = 'temp_file_name.jpg'
        ela_filename = 'temp_ela.png'
        image = Image.open(path).convert('RGB')
        image.save(temp_filename, 'JPEG', quality=quality)
        temp_image = Image.open(temp_filename)
        ela_image = ImageChops.difference(image, temp_image)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image


class CNNAnalysis:
    model = load_model('cnnfakedetectmodel.h5')

    def prepare_image(image_path):
        image_size = (128, 128)
        return np.array(ELAAnalysis.convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

    def cnn_model(image_path):
        mainimagepath = image_path
        image = CNNAnalysis.prepare_image(mainimagepath)
        img_arr = np.array(image)
        model = load_model('cnnfakedetectmodel.h5')
        # Expand the dimensions of the array
        img_arr = np.resize(img_arr, (128, 128, 3))
        x = np.expand_dims(img_arr, axis=0)
        # Make a prediction
        preds = model.predict(x)
        s = str(preds)
        s = s.replace('[', '')
        s = s.replace(']', '')
        sonuc = s.split(" ")
        for i in range(len(sonuc)):
            if sonuc[i] == '':
                del sonuc[i]
                break
        print(sonuc)
        results = ['sahte', 'gerçek']
        sonuc[0] = round(float(sonuc[0]), 4) * 100
        sonuc[1] = round(float(sonuc[1]), 4) * 100
        possibilities = [sonuc[0], sonuc[1]]
        palette_color = sns.color_palette('crest')
        sns.set_style("dark")
        plt.pie(possibilities, labels=results, colors=palette_color, autopct='%%%.2f',
                textprops={'fontsize': 12, 'fontfamily': 'Tahoma'})
        return plt

    def random_test(self):
        fake_image = os.listdir('casia-dataset/casia/CASIA2/testtp/')
        correctf = 0
        totalf = 0

        for file_name in fake_image:
            if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith(
                    'JPG') or file_name.endswith('bmp'):
                fake_image_path = os.path.join('casia-dataset/casia/CASIA2/testtp/', file_name)
                image = CNNAnalysis.prepare_image(fake_image_path)
                image = image.reshape(-1, 128, 128, 3)
                y_pred = CNNAnalysis.model.predict(image)
                y_pred_class = np.argmax(y_pred, axis=1)[0]
                totalf += 1

                if y_pred_class == 0:
                    correctf += 1
                    # print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
        print(f'Total: {totalf}, Correct: {correctf}, Acc: {correctf / totalf * 100.0}')

        real_image = os.listdir('casia-dataset/casia/CASIA2/testau/')
        correctg = 0
        totalg = 0
        for file_name in real_image:
            if file_name.endswith('jpg') or file_name.endswith('png') or file_name.endswith(
                    'JPG') or file_name.endswith('bmp'):
                real_image_path = os.path.join('casia-dataset/casia/CASIA2/testau/', file_name)
                image = CNNAnalysis.prepare_image(real_image_path)
                image = image.reshape(-1, 128, 128, 3)
                y_pred = CNNAnalysis.model.predict(image)
                y_pred_class = np.argmax(y_pred, axis=1)[0]
                totalg += 1
                if y_pred_class == 1:
                    correctg += 1
                    # print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

        correct = 0
        correct += correctf
        correct += correctg
        total = 0
        total += totalf
        total += totalg
        print("FAKELER:")
        print(f'Total: {totalf}, Correct: {correctf}, Acc: {correctf / totalf * 100.0}')
        print("GERÇEKLER:")
        print(f'Total: {totalg}, Correct: {correctg}, Acc: {correctg / totalg * 100.0}')
        print("GENEL:")
        print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')


def check_image_size(image_path):
    try:
        # Resim dosyasının boyutunu kontrol et
        file_size = os.stat(image_path).st_size
        if min_size <= file_size <= max_size:
            return True
        else:
            return False
    except Exception as e:
        print("Resim boyutu kontrol edilemedi: ", e)


def show_popup():
    popup = tk.Toplevel()
    popup.title("Dosya Boyutu Uygun Değil!")
    popup.geometry("400x50")
    width, height = 400, 50
    popup.geometry(f"{width}x{height}")
    x = (popup.winfo_screenwidth() - width) // 2
    y = (popup.winfo_screenheight() - height) // 2
    popup.geometry(f"+{x}+{y}")
    popup_label = tk.Label(popup, text="Yüklediğiniz dosyanın boyutu 50kb ile 5mb arasında olmalıdır.")
    popup_label.pack(padx=10, pady=10)


def image_fitter(image_path, maxh):
    image = Image.open(image_path)
    w, h = image.size
    if h > maxh:
        ratio = maxh / h
        image = image.resize((int(w * ratio), maxh))

    return image


def cnn_analysis_window():
    mainimagepath = grsl_directory.get()
    new_window3 = tk.Toplevel(root)
    new_window3.geometry(f"{width}x{height}")
    x = (new_window3.winfo_screenwidth() - width) // 2
    y = (new_window3.winfo_screenheight() - height) // 2
    new_window3.geometry(f"+{x}+{y}")
    new_window3.title("Yapay Zeka Analizi")
    cnn_plt = CNNAnalysis.cnn_model(mainimagepath)
    fig = cnn_plt.gcf()
    fig.clf()
    name = background_image
    maxh = 720
    imgbg = image_fitter(name, maxh)
    bg = ImageTk.PhotoImage(imgbg)
    lbl3 = Label(new_window3, image=bg)
    lbl3.place(x=0, y=0)
    # canvas
    canvas = tk.Canvas(new_window3, width=400, height=300, bg='white')
    canvas.pack(side=TOP, pady=25)
    canvas.create_text((200, 150))
    # image
    name = mainimagepath
    maxh = 300
    image = image_fitter(name, maxh)
    # Görseli canvas'a ekleme
    img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(200, 150, image=img)
    canvas.image = img
    cnn_plt = CNNAnalysis.cnn_model(mainimagepath)
    fig = cnn_plt.gcf()
    canvas2 = FigureCanvasTkAgg(fig, master=new_window3)
    canvas2.draw()
    canvas2.get_tk_widget().pack()
    new_window3.mainloop()
    new_window3.destroy()


def ela_analysis_window():
    mainimagepath = grsl_directory.get()
    new_window2 = tk.Toplevel(root)
    new_window2.geometry(f"{width}x{height}")
    x = (new_window2.winfo_screenwidth() - width) // 2
    y = (new_window2.winfo_screenheight() - height) // 2
    new_window2.geometry(f"+{x}+{y}")
    new_window2.title("ELA Analizi")
    name = background_image
    maxh = 720
    imgbg = image_fitter(name, maxh)
    bg = ImageTk.PhotoImage(imgbg)
    lbl2 = Label(new_window2, image=bg)
    lbl2.place(x=0, y=0)
    # canvas
    canvas = tk.Canvas(new_window2, width=400, height=300, bg='white')
    canvas.pack(side=TOP, pady=25)
    canvas.create_text((200, 150))
    # image
    name = mainimagepath
    maxh = 300
    image = image_fitter(name, maxh)
    # Görseli canvas'a ekleme
    img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(200, 150, image=img)
    canvas.image = img
    # canvas2
    canvas2 = tk.Canvas(new_window2, width=400, height=300, bg='white')
    canvas2.pack(side=TOP, pady=25)
    canvas2.create_text((200, 150))
    # image
    name = mainimagepath
    maxh = 300
    image = image_fitter(name, maxh)
    ela_image = ELAAnalysis.convert_to_ela_image(mainimagepath, 90)
    w, h = ela_image.size
    if h > 300:
        ratio = 300 / h
        ela_image = ela_image.resize((int(w * ratio), 300))
    # Görseli canvas'a ekleme
    img = ImageTk.PhotoImage(ela_image)
    canvas2.delete("all")
    canvas2.create_image(200, 150, image=img)
    canvas2.image = img
    ela = np.asarray(ela_image)
    ela_value = np.mean(ela)
    threshold = 200
    diff = np.amax(ela) - np.amin(ela)
    if diff < threshold:
        print("Resim sahte")
    else:
        print("Resim gerçek")
    new_window2.mainloop()
    new_window2.destroy()


def metadata_analysis_window():
    mainimagepath = grsl_directory.get()
    new_window = tk.Toplevel(root)
    width, height = 1280, 720
    new_window.geometry(f"{width}x{height}")
    x = (new_window.winfo_screenwidth() - width) // 2
    y = (new_window.winfo_screenheight() - height) // 2
    new_window.geometry(f"+{x}+{y}")
    new_window.title("Metadata Analizi")
    # görsel fitleme
    name = background_image
    maxh = 720
    imgbg = image_fitter(name, maxh)
    bg = ImageTk.PhotoImage(imgbg)
    lbl2 = Label(new_window, image=bg)
    lbl2.place(x=0, y=0)
    # canvas
    canvas = tk.Canvas(new_window, width=400, height=300, bg='white')
    canvas.pack(side=TOP, pady=5)
    canvas.create_text((200, 150))
    # image
    name = mainimagepath
    maxh = 300
    image = image_fitter(name, maxh)
    # Görseli canvas'a ekleme
    img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(200, 150, image=img)
    canvas.image = img
    # Create text widget and specify size.
    T = Text(new_window, height=20, width=52, relief="solid")
    # Create label
    l = Label(new_window, text="Exif Information")
    l.config(font=(14), bg="white")
    exif_result = MetadataAnalysis.get_exif(mainimagepath)
    s = str(exif_result)
    lineline = ""
    for i in range(len(s)):
        if s[i] == ",":
            lineline += "\n"
        elif s[i] == "'":
            lineline += ""
        elif s[i] == "{" or s[i] == "}":
            lineline += " "
        else:
            lineline += s[i]
    lineline += "\n"
    l.pack()
    T.pack()
    T.insert(tk.END, lineline)
    T.insert(tk.END, "\n\n " + MetadataAnalysis.analyze_metadata(mainimagepath))
    T.config(state=DISABLED)
    new_window.mainloop()


# URL'yi kullanarak görseli görüntüleme
def show_image_from_url():
    mainimagepath="downloaded.jpg"
    name = mainimagepath
    maxh = 300
    image = image_fitter(name, maxh)
    # Görseli canvas'a ekleme
    img = ImageTk.PhotoImage(image)
    canvas.delete("all")
    canvas.create_image(200, 150, image=img)
    canvas.image = img
    grsl_directory.insert(0,mainimagepath)


# URL'yi kullanarak görseli görüntüleme için fonksiyon çağırma
def download_image(url):
    filename = "downloaded.jpg"
    mainimagepath= os.getcwd()+filename
    with urllib.request.urlopen(url) as url_response:
        image_data = url_response.read()

    with open(filename, mode='wb') as file:
        file.write(image_data)


def show_image_from_url_entry():
    url = url_entry.get()
    download_image(url)
    show_image_from_url()


# Görseli seçmek için fonksiyon oluşturma
def select_image():
    # GÖRSEL SEÇ BUTONU İÇİN
    # Dosya seçme penceresini açma
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    mainimagepath = file_path
    if mainimagepath and check_image_size(mainimagepath):
        # Görseli açma ve boyutlandırma
        name = mainimagepath
        maxh = 300
        image = image_fitter(name, maxh)
        grsl_directory.delete(0, END)
        grsl_directory.insert(0, mainimagepath)
        # Görseli canvas'a ekleme
        img = ImageTk.PhotoImage(image)
        canvas.create_image(200, 150, image=img)
        canvas.image = img
        grsl_button.config(text="Başka bir görsel seç")
    else:
        show_popup()


def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
    # DİKDÖRTGEN YUVARLATICI
    points = [x1 + radius, y1, x1 + radius, y1, x2 - radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius, x2,
              y1 + radius, x2, y2 - radius, x2, y2 - radius, x2, y2, x2 - radius, y2, x2 - radius, y2, x1 + radius, y2,
              x1 + radius, y2, x1, y2, x1, y2 - radius, x1, y2 - radius, x1, y1 + radius, x1, y1 + radius, x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True)


# ANA PENCERE
root = tk.Tk()
root.title("Fake Photo Detection")
root.geometry("1280x720")
width, height = 1280, 720
root.geometry(f"{width}x{height}")
x = (root.winfo_screenwidth() - width) // 2
y = (root.winfo_screenheight() - height) // 2
root.geometry(f"+{x}+{y}")

# PENCERE BACKGROUNDU AYARLAMA
name = background_image
maxh = 720
imgbg = image_fitter(name, maxh)
bg = ImageTk.PhotoImage(imgbg)
lbl = Label(root, image=bg)
lbl.place(x=0, y=0)

# URL GİR label
url_label = tk.Label(root, text="URL girin:", bg="white", font="SF 10")
url_label.pack(side=TOP, pady=5)

# url combobox
url_entry = tk.Entry(root, width=65)
url_entry.pack(side=TOP, pady=5)

# URL ENTER BUTTON
url_button = tk.Button(root, text="URL'den Görsel Yükle", command=show_image_from_url_entry, bg="white")
url_button.pack(side=TOP, pady=5)

# GÖRSEL YÜKLE CANVASI
canvas = tk.Canvas(root, width=400, height=300, bg='white', highlightbackground="white")
canvas.pack(side=TOP, pady=5)
my_rectangle = round_rectangle(0, 0, 400, 300, radius=50, fill="#F6F6F6")
canvasimage = tk.PhotoImage(file=insert_image)
canvas.create_image(200, 150, anchor=tk.CENTER, image=canvasimage)

# görsel directory combobox
grsl_directory = tk.Entry(root, width=65)
grsl_directory.pack(side=TOP, pady=5)

# BUTONLAR
grsl_button = tk.Button(root, text="Görsel Seç", bg="white", command=lambda: select_image())
grsl_button.pack(side=TOP, pady=5)
metadata_button = tk.Button(root, text="Metadata\nAnalizi", bg="white", command=lambda: metadata_analysis_window())
metadata_button.place(x=530, y=480)
ela_button = tk.Button(root, text="ELA\nAnalizi", bg="white", command=lambda: ela_analysis_window())
ela_button.place(x=610, y=480)
cnn_button = tk.Button(root, text="Yapay Zeka\nAnalizi", bg="white", command=lambda: cnn_analysis_window())
cnn_button.place(x=675, y=480)
mainimagepath = grsl_directory.get()
root.mainloop()

root.destroy()
