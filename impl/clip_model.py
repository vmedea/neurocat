# SPDX-License-Identifier: MIT
import sys

import numpy

from .clip_util import normalize

MODEL_NAME = 'openai/clip-vit-large-patch14' # ViT-L/14

class TextModel:
    def __init__(self, device):
        '''Load text model.'''
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPProcessor, logging
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        logging.set_verbosity_error() # don't show annoying warning
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME)
        self.text_encoder.to(device)
        logging.set_verbosity_warning()

    def tokenize(self, text):
        text_input = self.tokenizer([text], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=False, return_tensors="pt")
        if len(text_input.input_ids[0]) != self.tokenizer.model_max_length:
            raise ValueError(f"Input too long ({len(text_input.input_ids[0])} > {self.tokenizer.model_max_length})")
        return text_input.input_ids

    def embedding_from_text(self, text):
        input_ids = self.tokenize(text)
        eot_token_pos = int(input_ids.argmax(dim=-1))

        text_embeddings = self.text_encoder(input_ids.to(self.text_encoder.device))
        return normalize(text_embeddings.text_embeds[0].detach().cpu().numpy())

class VisionModel:
    def __init__(self, device):
        ''''Load vision model.'''
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor, logging

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

        logging.set_verbosity_error() # don't show annoying warning
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME)
        self.image_encoder.to(device)
        logging.set_verbosity_warning()

    def embedding_from_image(self, image):
        from PIL import Image

        inputs = self.processor(images=image, return_tensors="pt")

        inputs.to(self.image_encoder.device)
        outputs = self.image_encoder(**inputs)
        return normalize(outputs.image_embeds[0].detach().cpu().numpy())
