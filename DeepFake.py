import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import PIL.Image
import cv2 as cv
import tempfile
import os
import yaml
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from tqdm import tqdm
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
from moviepy.editor import *

MYDIR = os.path.dirname(__file__) +"/"
content_image=""
style_image=""


def creaMp3(videoAudio,Audiovideo):
    audioclip = AudioFileClip(videoAudio)
    audioclip.write_audiofile("my_audio.mp3")
    clip = VideoFileClip(Audiovideo)
    videoclip = clip.set_audio(audioclip)
    videoclip.write_videofile("result2.mp4")


def combine_audio(vidname, audname): 
    video = VideoFileClip(audname)
    audio = video.audio
    audio.write_audiofile(vidname)
    return vidname 

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num




st.set_page_config(page_title="DEEPFAKE by I.A. Italia", page_icon=None, layout='wide', initial_sidebar_state='auto')

st.markdown("<center><h1> DEEPFAKE <small>by I. A. ITALIA</small></h1>", unsafe_allow_html=True)
st.write('<p style="text-align: center;font-size:15px;" >Crea <bold>Deep Fake </bold> divertenti e simpatici usando la nostra I.A.</bold><p>', unsafe_allow_html=True)


st.sidebar.subheader('\n\n Caricare la foto su cui applicare i movimenti del video')
selected_option = st.sidebar.file_uploader("Carica la tua immagine",type=["png","jpg","jpeg"],accept_multiple_files=False)

if (selected_option is not None) :
	st.sidebar.write("Foto caricata con successo...")
	with open(os.path.join(MYDIR,selected_option.name),"wb") as f:
		f.write(selected_option.getbuffer())
	#content_path = MYDIR + selected_option.name
	#content_image = load_img(content_path)
	image = Image.open(selected_option)
	img_array = np.array(image)
	st.sidebar.image(image)

st.sidebar.subheader('\n\nCarica il video da cui copiare i movimenti')
selected_option2 = st.sidebar.file_uploader("Carica la seconda immagine",type=["mp4"],accept_multiple_files=False)

if (selected_option2 is not None) :
	st.sidebar.write("Foto caricata con successo...")
	with open(os.path.join(MYDIR,selected_option2.name),"wb") as f:
		f.write(selected_option2.getbuffer())
	#style_path = MYDIR + selected_option2.name
	#style_image = load_img(style_path)
	
	video_file = open(MYDIR+selected_option2.name, 'rb')
	video_bytes = video_file.read()
	st.sidebar.video(video_bytes)
	
	
if (st.sidebar.button("Crea Deep Fake") ):
	source_image = imageio.imread(MYDIR + selected_option.name)
	reader = imageio.get_reader(MYDIR + selected_option2.name)
	fps = reader.get_meta_data()['fps']
	driving_video = []
	
	try:
		for im in reader:
			driving_video.append(im)
	except RuntimeError:
		pass
		
	reader.close()

	source_image = resize(source_image, (256, 256))[..., :3]
	driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
	generator, kp_detector = load_checkpoints(config_path="config/vox-adv-256.yaml", checkpoint_path="vox-adv-cpk.pth.tar", cpu=True)
	
	result_video ='result.mp4'
	relative="store_true"
	adapt_scale="store_true"
	find_best_frame="store_true"
	best_frame=None
	cpu="store_true"

	predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
	imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

	creaMp3(MYDIR + selected_option2.name, MYDIR + result_video)

	
	video_file2 = open(MYDIR + "result2.mp4", 'rb')
	video_bytes2 = video_file2.read()
	st.video(video_bytes2)
	
	os.remove(MYDIR + selected_option.name)
	os.remove(MYDIR + selected_option2.name)
	os.remove(MYDIR + result_video)
	
	

st.text("")
st.text("")
st.text("")
st.text("")
st.write("Proprietà intellettuale di [Intelligenza Artificiale Italia © ](https://intelligenzaartificialeitalia.net)")
st.write("Hai un idea e vuoi realizzare un Applicazione Web Intelligente? contatta il nostro [Team di sviluppatori © ](mailto:python.ai.solution@gmail.com)")
   

