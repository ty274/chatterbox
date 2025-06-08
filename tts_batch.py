import torchaudio as ta
import torch
import argparse
import os
from chatterbox.tts import ChatterboxTTS
import json
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import random


def main(args: argparse.Namespace) -> None:    
    """Main function to process text-to-speech generation."""
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    audio_files: Optional[Dict[str, List[Dict[str, Any]]]] = None
    speakers: Optional[List[str]] = None
    
    if args.audio_prompts is not None:
        try: 
            with open(args.audio_prompts, 'r') as f:
                audio_data: Any = json.load(f)
        except FileNotFoundError:
            print(f"Error: Audio prompts file '{args.audio_prompts}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load the JSON file '{args.audio_prompts}'.")
            sys.exit(1)

        if not isinstance(audio_data, list):
            print(f"The top level of {args.audio_prompts} must be a list.")
            sys.exit(1)

        # Organize the list by speakers. 
        # Audio files shorter than min_audio_len are excluded. 
        new_list: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for af in audio_data:
            if af['length'] < args.min_audio_len * 1000:
                continue

            new_list[af['speaker_id']].append(af)
        audio_files = new_list

        # Obtain a speaker list. 
        speakers = list(audio_files.keys())
        random.shuffle(speakers)

    # Automatically detect the best available device
    device: str
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load the model
    model: ChatterboxTTS = ChatterboxTTS.from_pretrained(device=device)

    # Create output directory if it doesn't exist
    os.makedirs(args.outputdir, exist_ok=True)

    # Read the text file and process each line
    try:
        with open(args.transcripts, 'r', encoding='utf-8') as file:
            lines: List[str] = file.readlines()
            
        # Filter out empty lines to get actual lines to process
        non_empty_lines: List[Tuple[int, str]] = [(i+1, line.strip()) for i, line in enumerate(lines) if line.strip()]
        total_files: int = len(non_empty_lines)
        
        print(f"Found {len(lines)} lines in {args.transcripts}")
        print(f"Will generate {total_files} audio files (skipping empty lines)")
        
        # Determine padding length based on total number of files
        padding_length: int = len(str(total_files))
        
        for file_index, (original_line_num, text) in enumerate(non_empty_lines, 1):
            print(f"Processing line {original_line_num}: {text[:50]}{'...' if len(text) > 50 else ''}")
                    
            # Generate speech from the line text
            if args.audio_prompts is not None and audio_files is not None and speakers is not None:
                spk: str = speakers[file_index % len(speakers)]
                audio_prompt: str = random.choice(audio_files[spk])['filename']
                wav: torch.Tensor = model.generate(text, audio_prompt_path=audio_prompt)
            else:
                wav = model.generate(text)
            
            # Create output filename with zero padding: 001.flac, 002.flac, etc.
            output_filename: str = f"{file_index:0{padding_length}d}.flac"
            output_path: str = os.path.join(args.outputdir, output_filename)
            
            # Save the generated audio
            ta.save(output_path, wav, model.sr, encoding="FLAC", bits_per_sample=16)
            print(f"Saved: {output_path}")

        print(f"Audio generation complete! Files saved in '{args.outputdir}' directory.")
        
    except FileNotFoundError:
        print(f"Error: Transcripts file '{args.transcripts}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parsing with comprehensive help messages."""
    parser = argparse.ArgumentParser(
        description='Generate high-quality speech audio from text using ChatterboxTTS. '
                   'Processes each line of a text file and creates numbered FLAC audio files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --transcripts dialogue.txt --outputdir ./audio_output
  %(prog)s -f script.txt -o ./speech -a speakers.json -t 5.0

Output files are named sequentially (001.flac, 002.flac, etc.) and saved as 
16-bit FLAC format for optimal quality and compatibility.
        ''')
    
    parser.add_argument('--transcripts', '-f', required=True, metavar='FILE',
                       help='Path to input text file containing lines to convert to speech. '
                           'Each non-empty line will generate a separate audio file. '
                           'Supports UTF-8 encoding for international characters.')
    
    parser.add_argument('--outputdir', '-o', required=True, metavar='DIR',
                       help='Directory where generated audio files will be saved. '
                           'Will be created automatically if it does not exist. '
                           'Files are saved as sequentially numbered FLAC files.')
    
    parser.add_argument('--audio_prompts', '-a', metavar='JSON_FILE',
                       help='Optional path to JSON file containing audio prompt catalog. '
                           'Enables voice cloning by cycling through different speaker voices. '
                           'JSON should contain a list of audio file metadata with speaker_id, '
                           'filename, and length fields.')
    
    parser.add_argument('--min_audio_len', '-t', type=float, default=6.0, metavar='SECONDS',
                       help='Minimum duration (in seconds) for audio files to be used as voice prompts. '
                           'Shorter audio clips are filtered out to ensure voice quality. '
                           'Default: 6.0 seconds. Only applies when --audio_prompts is used.')
    
    parser.add_argument('--seed', '-s', type=int, default=0, metavar='INT',
                       help='Random seed for reproducible results when using audio prompts. '
                           'Controls speaker selection and audio prompt randomization. '
                           'Default: 0.')    
    
    return parser


if __name__ == "__main__":
    parser: argparse.ArgumentParser = setup_parser()
    args: argparse.Namespace = parser.parse_args()
    main(args)