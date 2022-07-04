import numpy as np
import matplotlib.pyplot as plt
from envs.simple_point_bot import SimplePointBot
import pickle as pkl
import os
from skimage.transform import resize
# from modules import VanillaVAE
import torch
from train_trex_ens import Net
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x):
    if type(x) is not torch.Tensor and type(x) is not np.ndarray:
        x = np.array(x)
    if type(x) is not torch.Tensor:
        x = torch.FloatTensor(x)
    return x.to(TORCH_DEVICE)

env = SimplePointBot()

# os.mkdir('state_images_pb')
# os.chdir('state_images_pb')

# for i in range(0,180):
#    for j in range(0,150):
#        print(i, j)
#        img = env._draw_state([i,j])
#        img = resize(img, (64,64,3))
#        f = open("img_" + str(i) + "_" + str(j) + '.pkl', 'wb')
#        pkl.dump(img, f)
#        f.close()


# vae = VanillaVAE(params={'enc_lr':1e-4,'d_obs':(3,64,64), 'd_latent':8,'frame_stack':1,'enc_kl_multiplier':1e-6, 'enc_data_aug': False})
# vae.load_dict('/home/satvik/autolab/latent-space-safe-sets/latentsafesets/vae.pth')
ens = 5
for e in range(ens):
    # if ens == 0:
    #     continue
    model = Net()
    model.load_state_dict(torch.load('/home/satvik/autolab/latent-space-safe-sets/latentsafesets/trex_ens30_100_' + str(e) + ".pth"))
    model.eval()
    # reward_weights = [ 0.05076673,  0.30145349, -0.22433793 , 0.52192811 , 0.53506938,  0.00318889, -0.0896363, 0.5380223]
    rews = np.zeros((180,150))
    os.chdir('/home/satvik/autolab/latent-space-safe-sets/latentsafesets/state_images_pb')
    count = 0
    for img in os.listdir('/home/satvik/autolab/latent-space-safe-sets/latentsafesets/state_images_pb'):
        count += 1
        f = open(img,'rb')
        # embedding = vae.encode(torchify(pkl.load(f)))

        # reward = np.dot(reward_weights, embedding.cpu().numpy().flatten())
        img_arr = torchify(pkl.load(f)).unsqueeze(0)
        reward, _ = model.cum_return(img_arr)
        tokens = img.split("_")
        x = int(tokens[1]) #int(int(tokens[1])/10)
        y = int(tokens[2].replace('.pkl','')) #int(int(tokens[2].replace('.pkl',''))/10)
        print(count, " / ", 180*150)
        rews[x][y] = reward
        f.close()

    plt.imshow(rews)
    plt.clim(-0.1, 0.9)
    plt.colorbar()
    plt.savefig('../reward_heatmap30_100_' + str(e) + '.png')
    plt.close()