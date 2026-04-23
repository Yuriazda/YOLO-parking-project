import ultralytics
import matplotlib.pyplot as plt
import cv2
import os

model = ultralytics.YOLO("C:\\Users\\yurin\\Downloads\\teste\\runs\\detect\\train-6\\weights\\best.pt")

results = model.predict(
    "C:\\Users\\yurin\\Downloads\\teste\\data\\parking.png", 
    save=True, 
    conf=0.5,
    show_labels=False,  # remove os textos
    show_conf=False,    # remove as porcentagens
    line_width=1        # linhas mais finas
)

save_dir = str(results[0].save_dir)
files = os.listdir(save_dir)
img_path = os.path.join(save_dir, files[0])

img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.show()