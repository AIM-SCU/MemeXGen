#!/usr/bin/env python3
"""
LLaVA v1.6 13B Cross-Cultural Meme Translation Script
Enhanced with full checkpoint support and character rotation for variety
"""

import os
import sys
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LLaVAMemeTranslator:
    """LLaVA model wrapper for meme translation with character rotation"""
    
    # Define character batches for rotation
    CHARACTER_BATCHES = [
        "SpongeBob/Patrick/Squidward/Mr. Krabs",  # Batch 1: SpongeBob Universe (30%)
        "Homer Simpson/Peter Griffin/Rick and Morty/Bender",  # Batch 2: Adult Animation (25%)
        "Bugs Bunny/Daffy Duck/Tweety Bird/Roadrunner",  # Batch 3: Classic Looney Tunes (15%)
        "Tom and Jerry/Scooby-Doo/Pink Panther",  # Batch 4: Tom and Jerry + Friends (15%)
        "Mickey Mouse/Donald Duck/Goofy/Winnie the Pooh",  # Batch 5: Classic Disney (10%)
        "Adventure Time characters/Regular Show/Steven Universe/Gumball"  # Batch 6: Modern Cartoon Network (5%)
    ]
    
    def __init__(self, model_name: str = None):
        """Initialize the LLaVA model and processor"""
        self.model_name = self._get_model_path(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading LLaVA model: {self.model_name}")
        
        self.processor = None
        self.model = None
        self._load_model()
        
        # Base system prompt (without character suggestions)
        self.base_system_prompt = """ROLE: You are a cross-cultural meme specialist who creates NEW US-appropriate memes that capture the same emotional essence as the original.

INPUT: [Meme image] + [Cultural description] + [Original emotion/intensity (if provided, otherwise you must infer)]

TASK: If emotion and intensity are specified as "unknown", you must first analyze and determine them from the image and description. If they are provided, use those values.

OUTPUT FORMAT (all in English):

1. EMOTION ANALYSIS:
Primary Emotion: [joy/anger/sadness/fear/disgust] - (inferred from image/description if not provided, otherwise use given emotion)
Intensity Level: [low/high] - (inferred from image/description if not provided, otherwise use given intensity)  
Explanation: [Describe the core emotional trigger and why this intensity fits. If you inferred the emotion/intensity, explain your reasoning based on visual cues and cultural context]

2. CULTURAL ESSENCE:
The universal humor/feeling here is: [identify the core relatable human experience]. In US culture, this same feeling appears when [specific US scenario that evokes identical emotion].

3. IMAGE GENERATION INSTRUCTIONS:

Create a cartoon image using [CHARACTER_SUGGESTIONS]. Choose the most suitable character for this emotion. The character should be [specific positioning and gesture]. Background: [setting description]. Style: [maintain original character's art style]. Mood: [lighting/atmosphere]. NO speech bubbles or text.

4. US MEME CAPTION:
[Write ONLY the caption text here. Maximum 8 words. Make it punchy and memorable. No explanations. No quotation marks. No emojis. No hashtags. Just the pure text.]

CONSTRAINTS:
- Primary emotion: joy, anger, sadness, fear, or disgust only
- Intensity: low or high only  
- If emotion/intensity are "unknown", you MUST infer them from the image and description before proceeding
- Be creative - don't translate, create NEW content for US culture
- Visual instructions should be 25-35 words for clear narrative
- Image instructions must NOT include any dialogue, speech, or text elements
- Caption must be under 8 words maximum and highly readable
- Do not mix content between sections - keep each section separate and focused"""
        
        self.current_batch_index = 0
    
    def _get_character_suggestion(self, sample_index: int) -> str:
        """Get character suggestion based on sample index (rotating through batches)"""
        batch_index = sample_index % len(self.CHARACTER_BATCHES)
        return self.CHARACTER_BATCHES[batch_index]
    
    def _get_system_prompt(self, sample_index: int) -> str:
        """Generate system prompt with appropriate character suggestions"""
        character_suggestion = self._get_character_suggestion(sample_index)
        return self.base_system_prompt.replace("[CHARACTER_SUGGESTIONS]", character_suggestion)
    
    def _get_model_path(self, model_name: str) -> str:
        """Determine model path (local cache or online)"""
        if model_name:
            return model_name
            
        # Check for local cache
        local_cache_path = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/huggingface_cache/models--llava-hf--llava-v1.6-vicuna-13b-hf"
        
        if os.path.exists(local_cache_path):
            snapshots_dir = os.path.join(local_cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    snapshot_path = os.path.join(snapshots_dir, snapshots[0])
                    model_files = [f for f in os.listdir(snapshot_path) if f.endswith(('.bin', '.safetensors', '.json'))]
                    if model_files:
                        return snapshot_path
        
        return "llava-hf/llava-v1.6-vicuna-13b-hf"
    
    def _load_model(self):
        """Load the LLaVA model with optimized settings"""
        try:
            # Load processor
            self.processor = LlavaNextProcessor.from_pretrained(self.model_name, local_files_only=True)
            
            # Configure model loading
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": True,
                "local_files_only": True,
            }
            
            # Handle flash attention
            try:
                import flash_attn
            except ImportError:
                model_kwargs["attn_implementation"] = "eager"
            
            # Load model
            self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Local loading failed: {e}")
            logger.info("Falling back to online model...")
            
            try:
                # Fallback to online
                online_model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
                token = os.getenv('HF_TOKEN')
                
                if token:
                    self.processor = LlavaNextProcessor.from_pretrained(online_model_id, use_auth_token=token)
                else:
                    self.processor = LlavaNextProcessor.from_pretrained(online_model_id)
                
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "load_in_8bit": True,
                }
                
                if token:
                    model_kwargs["use_auth_token"] = token
                
                try:
                    import flash_attn
                except ImportError:
                    model_kwargs["attn_implementation"] = "eager"
                
                self.model = LlavaNextForConditionalGeneration.from_pretrained(online_model_id, **model_kwargs)
                self.model_name = online_model_id
                logger.info("Online model loaded successfully")
                
            except Exception as fallback_error:
                logger.error(f"Model loading failed completely: {fallback_error}")
                sys.exit(1)
    
    def translate_meme(self, image_path: str, content: str, emotion: str, intensity: str, sample_index: int = 0) -> str:
        """Translate a single meme using the model with character rotation"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get system prompt with appropriate character suggestions
            system_prompt = self._get_system_prompt(sample_index)
            
            # Handle emotion and intensity - check if they need to be inferred
            if emotion.lower() == "unknown" or intensity.lower() == "unknown":
                emotion_instruction = "Emotion and intensity are not provided - you must infer them from the image and description."
                emotion_text = "unknown (please infer)"
                intensity_text = "unknown (please infer)"
            else:
                emotion_instruction = "Use the provided emotion and intensity values."
                emotion_text = emotion
                intensity_text = intensity
            
            # Create prompt
            user_prompt = f"""Now analyze this meme:
Image: [PROVIDED]
Description: {content}
Original Emotion: {emotion_text}
Original Intensity: {intensity_text}

INSTRUCTION: {emotion_instruction}

Please follow the exact format with numbered sections."""
            
            # Prepare conversation
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt + "\n\n" + user_prompt},
                    {"type": "image", "image": image}
                ]
            }]
            
            # Process inputs
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Extract response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in response:
                generated_text = response.split("ASSISTANT:")[-1].strip()
            else:
                prompt_text = self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True)
                if prompt_text in response:
                    generated_text = response.replace(prompt_text, "").strip()
                else:  
                    generated_text = response.strip()
            
            return generated_text
            
        except Exception as e:
            return f"ERROR: {str(e)}"

class CheckpointManager:
    """Manages checkpoints for resuming interrupted runs"""
    
    def __init__(self, output_dir: Path, run_id: str = None):
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.run_id}.json"
        
    def save_checkpoint(self, processed_samples: List[Dict], remaining_samples: List[Dict], 
                       results: List[Dict], config: Dict):
        """Save current state to checkpoint"""
        checkpoint_data = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'processed_count': len(processed_samples),
            'remaining_count': len(remaining_samples),
            'total_count': len(processed_samples) + len(remaining_samples),
            'processed_samples': processed_samples,
            'remaining_samples': remaining_samples,
            'results': results,
            'config': config
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {self.checkpoint_file}")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Checkpoint loaded: {len(checkpoint_data['results'])} results, "
                           f"{len(checkpoint_data['remaining_samples'])} samples remaining")
                return checkpoint_data
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None
        return None
    
    def list_checkpoints(self) -> List[Path]:
        """List available checkpoints"""
        return list(self.checkpoint_dir.glob("checkpoint_*.json"))
    
    def cleanup_checkpoint(self):
        """Remove checkpoint file after successful completion"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleaned up")

class MemeTestSuite:
    """Test suite for meme translation with checkpoint support and character rotation"""
    
    def __init__(self, dataset_path: str, images_dir: str, output_dir: str = "test_results", 
                 model_path: str = None, run_id: str = None):
        """Initialize test suite"""
        self.dataset_path = dataset_path
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir, run_id)
        
        # Load dataset
        self.dataset = self._load_dataset()
        logger.info(f"Loaded {len(self.dataset)} entries from dataset")
        
        # Initialize model
        self.translator = LLaVAMemeTranslator(model_path)
        
        # Log character rotation info
        logger.info(f"Character rotation enabled with {len(LLaVAMemeTranslator.CHARACTER_BATCHES)} batches:")
        for i, batch in enumerate(LLaVAMemeTranslator.CHARACTER_BATCHES, 1):
            logger.info(f"  Batch {i}: {batch}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load the CSV dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)
    
    def _validate_sample(self, sample: Dict) -> Tuple[bool, str]:
        """Validate that a sample has required fields and image exists"""
        required_fields = ['filename', 'content', 'emotion', 'intensity']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in sample or not sample[field]]
        if missing_fields:
            return False, f"Missing fields: {missing_fields}"
        
        # Check image exists
        image_path = self.images_dir / sample['filename']
        if not image_path.exists():
            return False, f"Image not found: {image_path}"
        
        # Check image is valid
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
        
        # Check content length (allow shorter content since some might be brief)
        if len(sample['content'].strip()) < 5:
            return False, f"Content too short: '{sample['content']}'"
        
        return True, ""
    
    def select_random_samples(self, n: int = 300, seed: int = 42) -> List[Dict]:
        """Select n random samples from the dataset"""
        random.seed(seed)
        if n >= len(self.dataset):
            logger.info(f"Requested {n} samples, but dataset only has {len(self.dataset)} entries. Using all available samples.")
            return self.dataset
        
        selected = random.sample(self.dataset, n)
        logger.info(f"Selected {len(selected)} samples for testing")
        
        # Log emotion/intensity distribution in selected samples
        labeled_count = sum(1 for s in selected if s.get('emotion', '').lower() != 'unknown' and s.get('intensity', '').lower() != 'unknown')
        unknown_count = len(selected) - labeled_count
        logger.info(f"Distribution - Labeled: {labeled_count}, Unknown: {unknown_count}")
        
        return selected
    
    def resume_from_checkpoint(self) -> Optional[Tuple[List[Dict], List[Dict], Dict]]:
        """Check for and load existing checkpoint"""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            response = input(f"Found checkpoint with {len(checkpoint_data['results'])} completed samples. Resume? (y/n): ")
            if response.lower() == 'y':
                return (checkpoint_data['remaining_samples'], 
                       checkpoint_data['results'], 
                       checkpoint_data['config'])
        return None
    
    def run_test_batch(self, test_samples: List[Dict], resume_data: Optional[Tuple] = None) -> List[Dict]:
        """Run translation tests on a batch of samples with checkpoint support and character rotation"""
        
        # Handle resume
        if resume_data:
            remaining_samples, existing_results, config = resume_data
            logger.info(f"Resuming from checkpoint: {len(existing_results)} completed, {len(remaining_samples)} remaining")
            test_samples = remaining_samples
            results = existing_results
            # Calculate starting index for character rotation
            start_index = len(existing_results)
        else:
            results = []
            start_index = 0
        
        total_samples = len(test_samples) + len(results)
        
        logger.info(f"Processing {len(test_samples)} samples (Total run: {total_samples})...")
        
        # Progress tracking
        progress_interval = max(1, len(test_samples) // 20)
        checkpoint_interval = 25  # Save checkpoint every 25 samples
        
        # Track statistics
        valid_samples = 0
        labeled_samples = 0
        unknown_samples = 0
        
        # Track character batch usage
        batch_usage = {i: 0 for i in range(len(LLaVAMemeTranslator.CHARACTER_BATCHES))}
        
        processed_samples = []
        
        for i, sample in enumerate(test_samples, 1):
            is_valid, error_msg = self._validate_sample(sample)
            
            # Calculate absolute sample index for character rotation
            absolute_index = start_index + i - 1
            batch_index = absolute_index % len(LLaVAMemeTranslator.CHARACTER_BATCHES)
            
            if not is_valid:
                # Record skipped sample
                result = {
                    'filename': sample.get('filename', 'unknown'),
                    'original_content': sample.get('content', 'N/A'),
                    'original_emotion': sample.get('emotion', 'N/A'),
                    'original_intensity': sample.get('intensity', 'N/A'),
                    'translation': f"SKIPPED: {error_msg}",
                    'processing_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'skipped',
                    'data_type': 'invalid',
                    'character_batch': f"batch_{batch_index + 1}"
                }
                results.append(result)
                processed_samples.append(sample)
                continue
            
            valid_samples += 1
            
            # Track data type
            has_labels = (sample.get('emotion', '').lower() != 'unknown' and 
                         sample.get('intensity', '').lower() != 'unknown')
            if has_labels:
                labeled_samples += 1
                data_type = 'labeled'
            else:
                unknown_samples += 1
                data_type = 'unknown'
            
            # Track batch usage
            batch_usage[batch_index] += 1
            
            # Progress reporting
            current_total = len(results) + 1
            batch_name = f"Batch {batch_index + 1}"
            if i % progress_interval == 0 or i == len(test_samples):
                progress_pct = (current_total / total_samples) * 100
                print(f"Progress: {current_total}/{total_samples} ({progress_pct:.1f}%) - Processing: {sample['filename']} ({data_type}, {batch_name})")
            else:
                print(f"Processing {current_total}/{total_samples}: {sample['filename']} ({data_type}, {batch_name})")
            
            start_time = time.time()
            
            try:
                translation = self.translator.translate_meme(
                    str(self.images_dir / sample['filename']),
                    sample['content'],
                    sample['emotion'],
                    sample['intensity'],
                    absolute_index  # Pass the absolute index for character rotation
                )
                status = 'success'
            except Exception as e:
                translation = f"ERROR: Processing failed - {str(e)}"
                status = 'error'
                logger.warning(f"Failed to process {sample['filename']}: {str(e)}")
            
            processing_time = time.time() - start_time
            
            # Store result
            result = {
                'filename': sample['filename'],
                'original_content': sample['content'],
                'original_emotion': sample['emotion'],
                'original_intensity': sample['intensity'],
                'translation': translation,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'data_type': data_type,
                'character_batch': f"batch_{batch_index + 1}",
                'character_suggestions': LLaVAMemeTranslator.CHARACTER_BATCHES[batch_index]
            }
            
            results.append(result)
            processed_samples.append(sample)
            
            # Save checkpoint periodically
            if len(results) % checkpoint_interval == 0:
                remaining = test_samples[i:]
                config = {
                    'total_samples': total_samples,
                    'dataset_path': self.dataset_path,
                    'images_dir': str(self.images_dir),
                    'start_index': start_index
                }
                self.checkpoint_manager.save_checkpoint(processed_samples, remaining, results, config)
                
                # Also save intermediate results
                intermediate_file = self.save_results(results, f"_checkpoint_{len(results)}")
                logger.info(f"Checkpoint and intermediate results saved: {intermediate_file}")
        
        logger.info(f"Processed {valid_samples} valid samples")
        logger.info(f"Data distribution - Labeled: {labeled_samples}, Unknown: {unknown_samples}")
        logger.info(f"Character batch usage:")
        for batch_idx, count in batch_usage.items():
            if count > 0:
                logger.info(f"  Batch {batch_idx + 1}: {count} samples")
        
        return results
    
    def save_results(self, results: List[Dict], filename_suffix: str = "") -> str:
        """Save test results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"meme_translation_results_{timestamp}{filename_suffix}"
        
        # Save as CSV
        csv_path = self.output_dir / f"{base_filename}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Generate summary
        total_samples = len(results)
        successful = len([r for r in results if r.get('status') == 'success'])
        skipped = len([r for r in results if r.get('status') == 'skipped'])
        errors = len([r for r in results if r.get('status') == 'error'])
        labeled = len([r for r in results if r.get('data_type') == 'labeled'])
        unknown = len([r for r in results if r.get('data_type') == 'unknown'])
        
        if filename_suffix and "checkpoint" not in filename_suffix and "intermediate" not in filename_suffix:
            logger.info(f"Final results saved to: {csv_path}")
        logger.info(f"Summary - Total: {total_samples}, Success: {successful}, Skipped: {skipped}, Errors: {errors}")
        logger.info(f"Data types - Labeled: {labeled}, Unknown: {unknown}")
        
        return str(csv_path)

def main():
    """Main function with checkpoint support and character rotation"""
    # Configuration
    DATASET_PATH = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/data/emoji_labeled_data_train_synced.csv"
    IMAGES_DIR = "/WAVE/projects/oignat_lab/Yuming/memev1.0/llava13b/image/emo"
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='LLaVA meme translation with checkpoint support and character rotation')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, help='Dataset CSV path')
    parser.add_argument('--images', type=str, default=IMAGES_DIR, help='Images directory')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples (default: 300)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='test_results', help='Output directory')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID for checkpoint (auto-generated if not provided)')
    parser.add_argument('--resume', action='store_true', help='Force resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)
    
    if not os.path.exists(args.images):
        logger.error(f"Images directory not found: {args.images}")
        sys.exit(1)
    
    # Initialize test suite
    logger.info("Initializing LLaVA Meme Translation Test Suite with Character Rotation...")
    logger.info(f"Target samples: {args.samples}")
    test_suite = MemeTestSuite(args.dataset, args.images, args.output, args.model, args.run_id)
    
    # Check for resumption
    resume_data = None
    if args.resume or not args.run_id:
        resume_data = test_suite.resume_from_checkpoint()
    
    # Select samples (only if not resuming)
    if not resume_data:
        test_samples = test_suite.select_random_samples(args.samples, args.seed)
    else:
        test_samples = None  # Will be handled in run_test_batch
    
    # Run tests
    try:
        start_time = time.time()
        if resume_data:
            logger.info("Resuming from checkpoint...")
        else:
            logger.info(f"Starting fresh batch processing of {len(test_samples)} samples...")
        
        results = test_suite.run_test_batch(test_samples, resume_data)
        total_time = time.time() - start_time
        
        # Save final results
        results_file = test_suite.save_results(results, f"_final_n{len(results)}")
        
        # Clean up checkpoint
        test_suite.checkpoint_manager.cleanup_checkpoint()
        
        # Final summary with character batch statistics
        successful = len([r for r in results if r.get('status') == 'success'])
        labeled = len([r for r in results if r.get('data_type') == 'labeled'])
        unknown = len([r for r in results if r.get('data_type') == 'unknown'])
        avg_time_per_sample = total_time / len(results) if results else 0
        
        # Calculate batch distribution
        batch_counts = {}
        for r in results:
            batch = r.get('character_batch', 'unknown')
            batch_counts[batch] = batch_counts.get(batch, 0) + 1
        
        logger.info(f"=== FINAL RESULTS ===")
        logger.info(f"Total time: {total_time:.1f}s ({avg_time_per_sample:.1f}s per sample)")
        logger.info(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        logger.info(f"Data distribution - Labeled: {labeled}, Unknown: {unknown}")
        logger.info(f"Character batch distribution:")
        for batch in sorted(batch_counts.keys()):
            logger.info(f"  {batch}: {batch_counts[batch]} samples ({batch_counts[batch]/len(results)*100:.1f}%)")
        logger.info(f"Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        if 'results' in locals() and results:
            # Save partial results
            partial_file = test_suite.save_results(results, f"_interrupted_n{len(results)}")
            logger.info(f"Partial results saved to: {partial_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        if 'results' in locals() and results:
            # Save partial results on error
            error_file = test_suite.save_results(results, f"_error_n{len(results)}")
            logger.info(f"Partial results saved to: {error_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()
