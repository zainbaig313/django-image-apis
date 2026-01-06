from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse, HttpResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

#############################################
# ---------  FLOWER MODEL (LAZY) ----------
#############################################

flower_model = None

# Flower names
flower_names = [
    "pink primrose","hard-leaved pocket orchid","canterbury bells","sweet pea","wild geranium",
    "tiger lily","moon orchid","bird of paradise","monkshood","globe thistle","snapdragon",
    "colt's foot","king protea","spear thistle","yellow iris","globe-flower","purple coneflower",
    "peruvian lily","balloon flower","giant white arum lily","fire lily","pincushion flower",
    "fritillary","red ginger","grape hyacinth","corn poppy","prince of wales feathers",
    "stemless gentian","artichoke","sweet william","carnation","garden phlox","love in the mist",
    "mexican aster","alpine sea holly","ruby-lipped cattleya","cape flower","great masterwort",
    "siberian iris","lenten rose","barberton daisy","daffodil","sword lily","poinsettia","bolero deep blue",
    "wallflower","marigold","buttercup","oxeye daisy","common dandelion","petunia","wild pansy",
    "primula","sunflower","pelargonium","bishop of llandaff","gaura","geranium","orange dahlia",
    "pink-yellow dahlia","cautleya spicata","japanese anemone","black-eyed susan","silverbush",
    "californian poppy","osteospermum","spring crocus","iris","windflower","tree poppy","gazania",
    "azalea","water lily","rose","thorn apple","morning glory","passion flower","lotus","toad lily",
    "anthurium","frangipani","clematis","hibiscus","columbine","desert-rose","tree mallow","magnolia",
    "cyclamen","watercress","canna lily","hippeastrum","bee balm","pink quill","foxglove","bougainvillea",
    "camellia","mallow","mexican petunia","bromelia","blanket flower","trumpet creeper","blackberry lily",
    "common tulip","wild rose"
]

flower_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_flower_model():
    global flower_model
    if flower_model is None:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 102)
        state = torch.load("models/fine_tuned_resnet50.pth", map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        flower_model = model
    return flower_model


class FlowerPredictionAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)

        try:
            model = get_flower_model()

            image = Image.open(request.FILES['image']).convert("RGB")
            image_tensor = flower_transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                predicted = torch.argmax(output, dim=1).item()
                flower_name = flower_names[predicted]

            return JsonResponse({'flower': flower_name})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

# ===============================
# MOSAIC TRANSFORMER API
# ===============================

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        return self.conv2d(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        return x + residual

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=self.upsample)
        return self.conv2d(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

mosaic_model = None

def get_mosaic_model():
    global mosaic_model
    if mosaic_model is None:
        model = TransformerNet().to(device)
        state_dict = torch.load("models/mosaic.pth", map_location=device)

        # remove instance norm buffers if present
        for key in list(state_dict.keys()):
            if "running_mean" in key or "running_var" in key:
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        mosaic_model = model

    return mosaic_model


class MosaicAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if 'mosaic' not in request.FILES:
            return HttpResponse("No image provided", status=400)

        try:
            model = get_mosaic_model()

            img = Image.open(request.FILES['mosaic']).convert("RGB")

            transform_pipeline = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

            tensor = transform_pipeline(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor).cpu()

            out = out[0].clamp(0, 255).div(255)
            output_image = transforms.ToPILImage()(out)

            buf = io.BytesIO()
            output_image.save(buf, format="JPEG")
            buf.seek(0)

            return HttpResponse(buf, content_type="image/jpeg")

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)

# ===============================
# ANIME TRANSFORMER API
# ===============================

class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-9):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()
    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()
    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1)/float(n))
        scale_broadcast = self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        shift_broadcast = self.shift.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.refpad01_1 = nn.ReflectionPad2d(3)
        self.conv01_1 = nn.Conv2d(3, 64, 7)
        self.in01_1 = InstanceNormalization(64)
        self.conv02_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv02_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in02_1 = InstanceNormalization(128)
        self.conv03_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv03_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.in03_1 = InstanceNormalization(256)
        # ResBlocks
        self.refpad04_1 = nn.ReflectionPad2d(1)
        self.conv04_1 = nn.Conv2d(256, 256, 3)
        self.in04_1 = InstanceNormalization(256)
        self.refpad04_2 = nn.ReflectionPad2d(1)
        self.conv04_2 = nn.Conv2d(256, 256, 3)
        self.in04_2 = InstanceNormalization(256)
        self.refpad05_1 = nn.ReflectionPad2d(1)
        self.conv05_1 = nn.Conv2d(256, 256, 3)
        self.in05_1 = InstanceNormalization(256)
        self.refpad05_2 = nn.ReflectionPad2d(1)
        self.conv05_2 = nn.Conv2d(256, 256, 3)
        self.in05_2 = InstanceNormalization(256)
        self.refpad06_1 = nn.ReflectionPad2d(1)
        self.conv06_1 = nn.Conv2d(256, 256, 3)
        self.in06_1 = InstanceNormalization(256)
        self.refpad06_2 = nn.ReflectionPad2d(1)
        self.conv06_2 = nn.Conv2d(256, 256, 3)
        self.in06_2 = InstanceNormalization(256)
        self.refpad07_1 = nn.ReflectionPad2d(1)
        self.conv07_1 = nn.Conv2d(256, 256, 3)
        self.in07_1 = InstanceNormalization(256)
        self.refpad07_2 = nn.ReflectionPad2d(1)
        self.conv07_2 = nn.Conv2d(256, 256, 3)
        self.in07_2 = InstanceNormalization(256)
        self.refpad08_1 = nn.ReflectionPad2d(1)
        self.conv08_1 = nn.Conv2d(256, 256, 3)
        self.in08_1 = InstanceNormalization(256)
        self.refpad08_2 = nn.ReflectionPad2d(1)
        self.conv08_2 = nn.Conv2d(256, 256, 3)
        self.in08_2 = InstanceNormalization(256)
        self.refpad09_1 = nn.ReflectionPad2d(1)
        self.conv09_1 = nn.Conv2d(256, 256, 3)
        self.in09_1 = InstanceNormalization(256)
        self.refpad09_2 = nn.ReflectionPad2d(1)
        self.conv09_2 = nn.Conv2d(256, 256, 3)
        self.in09_2 = InstanceNormalization(256)
        self.refpad10_1 = nn.ReflectionPad2d(1)
        self.conv10_1 = nn.Conv2d(256, 256, 3)
        self.in10_1 = InstanceNormalization(256)
        self.refpad10_2 = nn.ReflectionPad2d(1)
        self.conv10_2 = nn.Conv2d(256, 256, 3)
        self.in10_2 = InstanceNormalization(256)
        self.refpad11_1 = nn.ReflectionPad2d(1)
        self.conv11_1 = nn.Conv2d(256, 256, 3)
        self.in11_1 = InstanceNormalization(256)
        self.refpad11_2 = nn.ReflectionPad2d(1)
        self.conv11_2 = nn.Conv2d(256, 256, 3)
        self.in11_2 = InstanceNormalization(256)
        self.deconv01_1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv01_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.in12_1 = InstanceNormalization(128)
        self.deconv02_1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv02_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.in13_1 = InstanceNormalization(64)
        self.refpad12_1 = nn.ReflectionPad2d(3)
        self.deconv03_1 = nn.Conv2d(64, 3, 7)

    def forward(self, x):
        y = F.relu(self.in01_1(self.conv01_1(self.refpad01_1(x))))
        y = F.relu(self.in02_1(self.conv02_2(self.conv02_1(y))))
        t04 = F.relu(self.in03_1(self.conv03_2(self.conv03_1(y))))
        t05 = self.in04_2(self.conv04_2(self.refpad04_2(F.relu(self.in04_1(self.conv04_1(self.refpad04_1(t04))))))) + t04
        t06 = self.in05_2(self.conv05_2(self.refpad05_2(F.relu(self.in05_1(self.conv05_1(self.refpad05_1(t05))))))) + t05
        t07 = self.in06_2(self.conv06_2(self.refpad06_2(F.relu(self.in06_1(self.conv06_1(self.refpad06_1(t06))))))) + t06
        t08 = self.in07_2(self.conv07_2(self.refpad07_2(F.relu(self.in07_1(self.conv07_1(self.refpad07_1(t07))))))) + t07
        t09 = self.in08_2(self.conv08_2(self.refpad08_2(F.relu(self.in08_1(self.conv08_1(self.refpad08_1(t08))))))) + t08
        t10 = self.in09_2(self.conv09_2(self.refpad09_2(F.relu(self.in09_1(self.conv09_1(self.refpad09_1(t09))))))) + t09
        t11 = self.in10_2(self.conv10_2(self.refpad10_2(F.relu(self.in10_1(self.conv10_1(self.refpad10_1(t10))))))) + t10
        t12 = self.in11_2(self.conv11_2(self.refpad11_2(F.relu(self.in11_1(self.conv11_1(self.refpad11_1(t11))))))) + t11
        y = F.relu(self.in12_1(self.deconv01_2(self.deconv01_1(t12))))
        y = F.relu(self.in13_1(self.deconv02_2(self.deconv02_1(y))))
        y = torch.tanh(self.deconv03_1(self.refpad12_1(y)))
        return y

anime_model = None

def get_anime_model():
    global anime_model
    if anime_model is None:
        model = Transformer().to(device)
        state = torch.load("models/shinkai_makoto.pth", map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        anime_model = model

    return anime_model


class AnimeAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if 'anime' not in request.FILES:
            return HttpResponse("No image provided", status=400)

        try:
            model = get_anime_model()

            img = Image.open(request.FILES['anime']).convert("RGB")

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1)
            ])

            t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(t)

            out = (out.squeeze(0).cpu() + 1) / 2
            out = transforms.ToPILImage()(out.clamp(0, 1))

            buf = io.BytesIO()
            out.save(buf, format="PNG")
            buf.seek(0)

            return HttpResponse(buf, content_type="image/png")

        except Exception as e:
            return HttpResponse(f"Error: {str(e)}", status=500)