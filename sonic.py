import os
import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, CLIPVisionModelWithProjection, AutoFeatureExtractor

from src.utils.util import save_videos_grid, seed_everything
from src.dataset.test_preprocess import process_bbox, image_audio_to_tensor
from src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, add_ip_adapters
from src.pipelines.pipeline_sonic import SonicPipeline
from src.models.audio_adapter.audio_proj import AudioProjModel
from src.models.audio_adapter.audio_to_bucket import Audio2bucketModel
from src.utils.RIFE.RIFE_HDv3 import RIFEModel
from src.dataset.face_align.align import AlignImage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test(pipe, config, whisper, audio2token, audio2bucket, image_encoder, width, height, batch, device_audio, device_encoder, device_main):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device_main).float()

    ref_img = batch['ref_img']
    clip_img = batch['clip_images'].to(device_encoder)
    face_mask = batch['face_mask']

    image_encoder.to(device_encoder)  # 确保模型在cuda:1
    clip_img = clip_img.to(device_encoder)  # 确保数据也在cuda:1
    image_embeds = image_encoder(clip_img).image_embeds

    audio_feature = batch['audio_feature'].to(device_audio)
    audio_len = batch['audio_len']
    step = int(config.step)

    window = 3000
    audio_prompts = []
    last_audio_prompts = []
    for i in range(0, audio_feature.shape[-1], window):
        audio_prompt = whisper.encoder(audio_feature[:, :, i:i+window], output_hidden_states=True).hidden_states
        last_audio_prompt = whisper.encoder(audio_feature[:, :, i:i+window]).last_hidden_state
        last_audio_prompt = last_audio_prompt.unsqueeze(-2)
        audio_prompt = torch.stack(audio_prompt, dim=2)
        audio_prompts.append(audio_prompt)
        last_audio_prompts.append(last_audio_prompt)

    audio_prompts = torch.cat(audio_prompts, dim=1)[:, :audio_len*2]
    audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:, :4]), audio_prompts, torch.zeros_like(audio_prompts[:, :6])], dim=1)

    last_audio_prompts = torch.cat(last_audio_prompts, dim=1)[:, :audio_len*2]
    last_audio_prompts = torch.cat([torch.zeros_like(last_audio_prompts[:, :24]), last_audio_prompts, torch.zeros_like(last_audio_prompts[:, :26])], dim=1)

    ref_tensor_list = []
    audio_tensor_list = []
    uncond_audio_tensor_list = []
    motion_buckets = []
    for i in tqdm(range(audio_len // step)):
        audio_clip = audio_prompts[:, i*2*step:i*2*step+10].unsqueeze(0).to(device_audio)
        audio_clip_for_bucket = last_audio_prompts[:, i*2*step:i*2*step+50].unsqueeze(0).to(device_audio)

        motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds.to(device_audio))
        motion_bucket = motion_bucket * 16 + 16
        motion_buckets.append(motion_bucket[0].to(device_main))

        cond_audio_clip = audio2token(audio_clip).squeeze(0).to(device_main)
        uncond_audio_clip = audio2token(torch.zeros_like(audio_clip)).squeeze(0).to(device_main)

        ref_tensor_list.append(ref_img[0])
        audio_tensor_list.append(cond_audio_clip[0])
        uncond_audio_tensor_list.append(uncond_audio_clip[0])

    video = pipe(
        ref_img,
        clip_img,
        face_mask,
        audio_tensor_list,
        uncond_audio_tensor_list,
        motion_buckets,
        height=height,
        width=width,
        num_frames=len(audio_tensor_list),
        decode_chunk_size=config.decode_chunk_size,
        motion_bucket_scale=config.motion_bucket_scale,
        fps=config.fps,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale1=config.min_appearance_guidance_scale,
        max_guidance_scale1=config.max_appearance_guidance_scale,
        min_guidance_scale2=config.audio_guidance_scale,
        max_guidance_scale2=config.audio_guidance_scale,
        overlap=config.overlap,
        shift_offset=config.shift_offset,
        frames_per_batch=config.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength
    ).frames

    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = torch.cat([video.to(device_main)], dim=0).cpu()
    return video

class Sonic():
    config_file = os.path.join(BASE_DIR, 'config/inference/sonic.yaml')
    config = OmegaConf.load(config_file)

    def __init__(self, device_id=0, enable_interpolate_frame=True):
        config = self.config
        config.use_interframe = enable_interpolate_frame

        self.device_0 = torch.device("cuda:0")
        self.device_1 = torch.device("cuda:1")

        config.pretrained_model_name_or_path = os.path.join(BASE_DIR, config.pretrained_model_name_or_path)

        vae = AutoencoderKLTemporalDecoder.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae", variant="fp16").to(self.device_0)
        scheduler = EulerDiscreteScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16").to(self.device_1)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet", variant="fp16").to(self.device_0)
        add_ip_adapters(unet, [32], [config.ip_audio_scale])

        audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024, context_tokens=32).to(self.device_1)
        audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024, output_dim=1, context_tokens=2).to(self.device_1)

        unet.load_state_dict(torch.load(os.path.join(BASE_DIR, config.unet_checkpoint_path), map_location=self.device_0), strict=True)
        audio2token.load_state_dict(torch.load(os.path.join(BASE_DIR, config.audio2token_checkpoint_path), map_location=self.device_1), strict=True)
        audio2bucket.load_state_dict(torch.load(os.path.join(BASE_DIR, config.audio2bucket_checkpoint_path), map_location=self.device_1), strict=True)

        whisper = WhisperModel.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/')).to(self.device_0).eval()
        whisper.requires_grad_(False)
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(BASE_DIR, 'checkpoints/whisper-tiny/'))
        self.face_det = AlignImage(self.device_0, det_path=os.path.join(BASE_DIR, 'checkpoints/yoloface_v5m.pt'))

        if config.use_interframe:
            rife = RIFEModel(device=self.device_0)
            rife.load_model(os.path.join(BASE_DIR, 'checkpoints', 'RIFE/'))
            self.rife = rife

        pipe = SonicPipeline(unet=unet, image_encoder=image_encoder, vae=vae, scheduler=scheduler)
        pipe = pipe.to(device=self.device_0, dtype=torch.float16)

        self.pipe = pipe
        self.whisper = whisper
        self.audio2token = audio2token
        self.audio2bucket = audio2bucket
        self.image_encoder = image_encoder
        self.device = self.device_0
        self.device_encoder = self.device_1
        self.device_audio = self.device_1
        self.device_whisper = self.device_0

        print('init done with multi-GPU')

    def preprocess(self, image_path, expand_ratio=1.0):
        face_image = cv2.imread(image_path)
        h, w = face_image.shape[:2]
        _, _, bboxes = self.face_det(face_image, maxface=True)
        face_num = len(bboxes)
        bbox = []
        if face_num > 0:
            x1, y1, ww, hh = bboxes[0]
            x2, y2 = x1 + ww, y1 + hh
            bbox = x1, y1, x2, y2
            bbox_s = process_bbox(bbox, expand_radio=expand_ratio, height=h, width=w)

        return {'face_num': face_num, 'crop_bbox': bbox_s}

    def crop_image(self, input_image_path, output_image_path, crop_bbox):
        face_image = cv2.imread(input_image_path)
        crop_image = face_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        cv2.imwrite(output_image_path, crop_image)

    @torch.no_grad()
    def process(self, image_path, audio_path, output_path, min_resolution=512, inference_steps=25, dynamic_scale=1.0, keep_resolution=False, seed=None):
        config = self.config
        seed_everything(seed if seed else config.seed)

        config.num_inference_steps = inference_steps
        config.motion_bucket_scale = dynamic_scale

        video_path = output_path.replace('.mp4', '_noaudio.mp4')
        audio_video_path = output_path

        imSrc_ = Image.open(image_path).convert('RGB')
        raw_w, raw_h = imSrc_.size

        test_data = image_audio_to_tensor(self.face_det, self.feature_extractor, image_path, audio_path, limit=config.frame_num, image_size=min_resolution, area=config.area)
        if test_data is None:
            return -1

        for k, v in test_data.items():
            if isinstance(v, torch.Tensor):
                test_data[k] = v.unsqueeze(0)

        test_data['ref_img'] = test_data['ref_img'].to(self.device)
        test_data['clip_images'] = test_data['clip_images'].to(self.device_encoder)
        test_data['face_mask'] = test_data['face_mask'].to(self.device)
        test_data['audio_feature'] = test_data['audio_feature'].to(self.device_whisper)

        height, width = test_data['ref_img'].shape[-2:]
        resolution = f'{raw_w//2*2}x{raw_h//2*2}' if keep_resolution else f'{width}x{height}'

        from sonic import test
        video = test(
            self.pipe,
            config,
            self.whisper,
            self.audio2token,
            self.audio2bucket,
            self.image_encoder,
            width,
            height,
            test_data,
            self.device_audio,
            self.device_encoder,
            self.device
        )
        if config.use_interframe:
            out = video.to(self.device)
            results = []
            for idx in tqdm(range(out.shape[2]-1), ncols=0):
                I1, I2 = out[:, :, idx], out[:, :, idx+1]
                middle = self.rife.inference(I1, I2).clamp(0, 1).detach()
                results.extend([I1, middle])
            results.append(out[:, :, -1])
            video = torch.stack(results, 2).cpu()

        save_videos_grid(video, video_path, n_rows=video.shape[0], fps=config.fps * 2 if config.use_interframe else config.fps)
        os.system(f'ffmpeg -i "{video_path}" -i "{audio_path}" -s {resolution} -vcodec libx264 -acodec aac -crf 18 -shortest -y "{audio_video_path}"')
        os.remove(video_path)
        return 0
