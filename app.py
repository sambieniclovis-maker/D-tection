import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES_VOC = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

st.setTitle('Mon Détecteur d’Objets')
st.write('Glissez-déposez une image pour tester le modèle.')

# Chargement du modèle (assurez-vous d'avoir votre fichier mon_modele.pth)
# model = VotreArchitecture()
# model.load_state_dict(torch.load('mon_modele.pth', map_location=device))
model.to(device)
model.eval()

# Widget pour uploader l'image
uploaded_file = st.file_uploader(
    'Choisissez une image...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  img_pil = Image.open(uploaded_file).convert('RGB')

  # Inférence
  transform = T.Compose([T.ToTensor()])
  img_tensor = transform(img_pil).to(device)

  with torch.no_grad():
    predictions = model([img_tensor])

  pred = predictions[0]
  pred_boxes = pred['boxes'].cpu()
  pred_labels = pred['labels'].cpu()
  pred_scores = pred['scores'].cpu()

  seuil = 0.5
  mask = pred_scores >= seuil
  pred_boxes = pred_boxes[mask]
  pred_labels = pred_labels[mask]
  pred_scores = pred_scores[mask]

  # Dessin
  draw = ImageDraw.Draw(img_pil)
  if len(pred_boxes) > 0:
    for box, label_idx, score in zip(pred_boxes, pred_labels, pred_scores):
      xmin, ymin, xmax, ymax = box.tolist()
      idx = label_idx.item()
      nom_classe = CLASSES_VOC[idx] if idx < len(CLASSES_VOC) else f'Classe {idx}'
      texte = f'{nom_classe} ({score.item()*100:.1f}%)'

      draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=3)
      draw.text((xmin, max(0, ymin - 15)), texte, fill='red')

  # Affichage sur la page web
  st.image(img_pil, caption='Image traitée', use_column_width=True)
