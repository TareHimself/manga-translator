import numpy as np
import math
import cv2
import random
import torch
import torch.nn as nn
import os
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from .datasets import ColorDetectionDataset
from .models import get_color_detection_model


def train_model(num_samples=6000,num_workers=5,backgrounds=[],epochs = 1000,batch_size=32,learning_rate=0.001,weights_path=None,seed = 200,train_device=torch.device("cuda:0")):

    dataset = ColorDetectionDataset(generate_target=num_samples,backgrounds=backgrounds,generator_seed=seed,num_workers=num_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

#efficientformerv2_s2.snap_dist_in1k
#resnet18
# model = timm.create_model("efficientformerv2_s2.snap_dist_in1k",pretrained=True,in_chans=3,num_classes=3).to(PYTORCH_DEVICE)
    model = get_color_detection_model(weights_path=weights_path)
    model = model.to(train_device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    best_loss = 99999999999
    best_epoch = 0
    patience = 20
    patience_count = 0

    best_state = copy.deepcopy(model.state_dict())

    try:
        for epoch in range(epochs):
            loading_bar = tqdm(total=len(dataloader))
            total_accu, total_count = 0, 0
            running_loss = 0
            for idx, data in enumerate(dataloader):
                images, results = data
                images = images.type(torch.FloatTensor).to(train_device)
                results = results.type(torch.FloatTensor).to(train_device)

                optimizer.zero_grad()

                outputs: torch.Tensor = model(images)

                loss = criterion(outputs, results)

                loss.backward()
                optimizer.step()

                # Calculate the element-wise distance between the arrays
                distance = torch.abs(results - outputs)

                # Set a threshold value for considering the elements as accurate
                threshold = 0.1

                # Calculate the accuracy
                total_accu += (distance <= threshold).float().mean().item() * results.size(0)
                total_count += results.size(0)
                running_loss += loss.item() * images.size(0)
                loading_bar.set_description_str(
                    f'Training | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(dataloader.dataset)):.8f} | ')
                loading_bar.update()

            loading_bar.set_description_str( f'Training | epoch {epoch + 1}/{epochs} |  Accuracy {((total_accu / total_count) * 100):.4f} | loss {(running_loss / len(dataloader.dataset)):.8f} | ')
            loading_bar.close()
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accu = (total_accu / total_count) * 100
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            
    except KeyboardInterrupt:
        print("Stopping training early at user request")
    
    print("Best epoch at ",best_epoch,"with loss",best_loss)
    model.load_state_dict(best_state)
    return model




# with torch.no_grad():
#     with torch.inference_mode():
#         test_model = get_color_detection_model("state.pt")
#         test_model = test_model.to(PYTORCH_DEVICE)
#         test_model.eval()
#         for test in [cv2.imread("Text Drawn.png"),cv2.imread("Frame Section.png"),cv2.imread('test_1.png'),cv2.imread('test_2.png')]:
#             try:
#                 # frame,_ = generate_example("Test",test_bg,size=(200,200))
#                 frame = test
#                 to_eval = frame.copy()
#                 to_eval = prep_sample(to_eval).unsqueeze(0).type(torch.FloatTensor).to(PYTORCH_DEVICE)
#                 results = test_model(to_eval)[0]
#                 color = np.array(results.cpu().numpy() * 255,dtype=np.int32)
#                 print("Detected color",color)
#                 debug_image(frame,"Test Frame")
#             except KeyboardInterrupt:
#                 break

#             generator = random.Random(20)
#         while True:
#             try:
#                 # frame,_ = generate_example("Test",test_bg,size=(200,200))
#                 frame,truth = generate_example("Test",background=generator.choice([generator.choice(backgrounds),np.ones((*dataset.generate_size,3),dtype=np.uint8) * 255,np.ones((*dataset.generate_size,3),dtype=np.uint8) * 0]),size=dataset.generate_size,font_file=generator.choice(["fonts/animeace2_reg.ttf","fonts/BlambotClassicBB.ttf"]),generator=generator)
#                 to_eval = frame.copy()
#                 to_eval = prep_sample(to_eval).unsqueeze(0).type(torch.FloatTensor).to(PYTORCH_DEVICE)
#                 results = test_model(to_eval)[0]
#                 color = np.array(results.cpu().numpy() * 255,dtype=np.int32)
#                 print("Detected color",color)
#                 print("Actual color",truth)
#                 debug_image(frame,"Test Frame")
#             except KeyboardInterrupt:
#                 break


