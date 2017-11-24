import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import keys

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = keys.alphabet
nclass = len(alphabet) + 1
print nclass

model = crnn.CRNN(32, 1, nclass, 256)
print('loading pretrained model from %s' % model_path)
state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
