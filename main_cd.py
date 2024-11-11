import torch
from translator.color_detect.train import train_model

model = train_model(num_samples=2000)

#model = model.to(torch.device('cpu'))


# gen = SampleGenerator()
# gen.run(5000,seed=2)
# gen.run(5000,seed=5)