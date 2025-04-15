from PIL import Image
import io
import numpy as np
import torch

def clip_compressibility_combined(clip_weight=0.7, compress_weight=0.3):
    """
    Combined reward function that balances CLIP-based image-text alignment with JPEG compressibility.
    
    Args:
        clip_weight: Weight for the CLIP score component (default: 0.7)
        compress_weight: Weight for the compressibility component (default: 0.3)
    
    Returns a reward function that:
    1. Evaluates how well images match their animal prompts using CLIP
    2. Measures image compressibility using JPEG compression
    3. Combines both rewards with the specified weighting
    """
    import torch
    import numpy as np
    import io
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel
    
    # Initialize CLIP model for image-text similarity
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Normalize weights to ensure they sum to 1
    total = clip_weight + compress_weight
    clip_weight = clip_weight / total
    compress_weight = compress_weight / total
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # Convert to PIL for JPEG compression
        pil_images = [Image.fromarray(image) for image in images]
        
        # 1. JPEG Compressibility component
        buffers = [io.BytesIO() for _ in pil_images]
        for image, buffer in zip(pil_images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        
        # Get file sizes in KB - smaller is better for compressibility
        sizes = np.array([buffer.tell() / 1000 for buffer in buffers])
        
        # 3. Normalize the compressibility scores to 0-1 range
        max_size = 100.0  # KB - use this as our reference point
        compress_scores = np.clip(1.0 - (sizes / max_size), 0.0, 1.0)
        
        # 2. CLIP score component - extract animal names from prompts
        animal_terms = []
        for i, prompt in enumerate(prompts):
            # Extract the base animal name from metadata if available
            if isinstance(metadata, list) and i < len(metadata) and 'base_animal' in metadata[i]:
                animal_terms.append(f"a {metadata[i]['base_animal']}")
            else:
                # Fallback to full prompt
                animal_terms.append(prompt)
        
        # Process with CLIP
        with torch.no_grad():
            inputs = processor(text=animal_terms, images=pil_images, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get features and calculate similarity
            image_features = model.get_image_features(inputs['pixel_values'])
            text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity scores
            clip_scores = torch.sum(image_features * text_features, dim=1).cpu().numpy()
        
        # Clip scores are already in a good range (0-1) from cosine similarity
        clip_scores = np.clip(clip_scores, 0, 1)
        
        # 4. Combine the rewards with weighting
        combined_scores = (clip_weight * clip_scores) + (compress_weight * compress_scores)
        
        # Return combined scores and metadata for analysis
        return combined_scores, {
            "clip_scores": clip_scores.tolist(),
            "compress_scores": compress_scores.tolist(),
            "animal_terms": animal_terms
        }
    
    return _fn

def clip_score_animal_only():
    import torch
    import numpy as np
    from transformers import CLIPProcessor, CLIPModel
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # Use the base animal names from metadata instead of the full prompts
        animal_prompts = [f"a {meta['base_animal']}" for meta in metadata]
        
        with torch.no_grad():
            # Process with animal-only prompts
            inputs = processor(text=animal_prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get image and text features
            image_features = model.get_image_features(inputs['pixel_values'])
            text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarities = torch.sum(image_features * text_features, dim=1)
            
            # Ensure scores are between 0 and 1
            scores = torch.clamp(similarities, min=0, max=1).cpu().numpy()
            
        return np.array(scores), {"animal_prompts": animal_prompts}
    
    return _fn

def clip_score():
    import torch
    import numpy as np
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # Process images and prompts in batches
        with torch.no_grad():
            # Prepare inputs for the model
            inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Get image and text features
            image_features = model.get_image_features(inputs['pixel_values'])
            text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarities = torch.sum(image_features * text_features, dim=1)
            
            # Ensure scores are between 0 and 1
            scores = torch.clamp(similarities, min=0, max=1).cpu().numpy()
            
        return np.array(scores), {}
    
    return _fn

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }
        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"]).squeeze()
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            all_info["precision"] += (
                np.array(response_data["precision"]).squeeze().tolist()
            )
            all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
