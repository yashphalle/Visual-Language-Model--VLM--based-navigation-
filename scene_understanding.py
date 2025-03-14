import cv2
import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-Ti/16", device=device)

text_prompts = ["normal scene", "violent scene"]
text_tokens = clip.tokenize(text_prompts).to(device)


ATTACK_THRESHOLD = 0.5


video_source = 0  
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame (BGR to RGB) and prepare PIL image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    input_image = preprocess(pil_img).unsqueeze(0).to(device)

    # Get image features and compute similarity with text features
    with torch.no_grad():
        image_features = model.encode_image(input_image)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Compute similarity score between image and each text prompt
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # We assume index 1 corresponds to "violent scene"
        attack_prob = similarity[0, 1].item()

    # Determine label based on threshold
    label = "ATTACK" if attack_prob > ATTACK_THRESHOLD else "Normal"
    color = (0, 0, 255) if label == "ATTACK" else (0, 255, 0)

    # Overlay the label and score on the frame
    cv2.putText(frame, f"{label}: {attack_prob:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Scene Understanding Demo", frame)

    # Press 'q' to quit the demo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
