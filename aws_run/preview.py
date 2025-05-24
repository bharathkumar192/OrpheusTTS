import os
import glob
import pandas as pd
import argparse
from pathlib import Path
import json
from collections import defaultdict, Counter
import re

# Same configuration as the main script
SPEAKERS = [
    "aisha", "anika", "arfa", "asmr", "nikita", 
    "raju", "rhea", "ruhaan", "sangeeta", "shayana"
]

HINDI_SENTENCES = [
    "ब्रिलियंट आइडिया! कस्टमर फीडबैक को प्रोडक्ट डेवलपमेंट साइकिल में इंटीग्रेट करना बहुत ज़रूरी है...",
    "रात को हाईवे पर ड्राइव करना है, तो विज़िबिलिटी के लिए कुछ ब्राइट कलर का पहनना अच्छा आइडिया नहीं है...",
    "तुम्हे क्या लगता है, क्या आर्टिफिशियल इंटेलिजेंस फ्यूचर में ह्यूमन क्रिएटिविटी को रिप्लेस कर सकता है...",
    "अगर तुम अपनी पब्लिक स्पीकिंग स्किल्स इम्प्रूव करना चाहते हो, तो टोस्टमास्टर्स क्लब जॉइन करना काफी फायदेमंद हो सकता है...",
    "यार, इस प्रोजेक्ट में इतना स्ट्रेस क्यों है भाई! डेडलाइन तो अभी भी 2 दिन बाकी है, चिल मारो यार..."
]

def extract_metadata_from_filename(filepath):
    """Extract metadata from the filename and path structure."""
    path_parts = Path(filepath).parts
    filename = Path(filepath).stem
    
    checkpoint = None
    speaker = None
    sentence_idx = None
    
    for part in path_parts:
        if part.startswith('checkpoint-'):
            checkpoint = part
        elif part in SPEAKERS:
            speaker = part
    
    # Extract sentence index from filename
    match = re.search(r'sentence_(\d+)', filename)
    if match:
        sentence_idx = int(match.group(1))
    
    return checkpoint, speaker, sentence_idx

def get_file_size_mb(filepath):
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except:
        return 0

def analyze_directory(input_dir):
    """Analyze the batch inference results directory."""
    print(f"🔍 Analyzing directory: {input_dir}")
    print("=" * 60)
    
    # Find all WAV files
    wav_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    if not wav_files:
        print("❌ No WAV files found!")
        return None
    
    print(f"📁 Found {len(wav_files)} audio files")
    
    # Analyze file structure
    analysis = {
        'files': [],
        'checkpoints': defaultdict(int),
        'speakers': defaultdict(int),
        'sentences': defaultdict(int),
        'file_sizes': [],
        'total_size': 0,
        'missing_metadata': [],
        'valid_files': 0
    }
    
    for wav_file in wav_files:
        checkpoint, speaker, sentence_idx = extract_metadata_from_filename(wav_file)
        file_size = get_file_size_mb(wav_file)
        
        file_info = {
            'path': wav_file,
            'filename': os.path.basename(wav_file),
            'checkpoint': checkpoint,
            'speaker': speaker,
            'sentence_idx': sentence_idx,
            'size_mb': file_size,
            'valid': bool(checkpoint and speaker and sentence_idx)
        }
        
        analysis['files'].append(file_info)
        analysis['file_sizes'].append(file_size)
        analysis['total_size'] += file_size
        
        if file_info['valid']:
            analysis['checkpoints'][checkpoint] += 1
            analysis['speakers'][speaker] += 1
            analysis['sentences'][sentence_idx] += 1
            analysis['valid_files'] += 1
        else:
            analysis['missing_metadata'].append(wav_file)
    
    return analysis

def print_analysis_report(analysis):
    """Print a detailed analysis report."""
    print("\n📊 ANALYSIS REPORT")
    print("=" * 60)
    
    # Basic stats
    print(f"Total Files: {len(analysis['files'])}")
    print(f"Valid Files: {analysis['valid_files']}")
    print(f"Invalid Files: {len(analysis['missing_metadata'])}")
    print(f"Total Size: {analysis['total_size']:.1f} MB")
    print(f"Average File Size: {sum(analysis['file_sizes'])/len(analysis['file_sizes']):.2f} MB")
    
    # Checkpoints breakdown
    print(f"\n🔄 CHECKPOINTS ({len(analysis['checkpoints'])})")
    expected_per_checkpoint = 10 * 5  # 10 speakers × 5 sentences
    for checkpoint in sorted(analysis['checkpoints'].keys()):
        count = analysis['checkpoints'][checkpoint]
        status = "✅" if count == expected_per_checkpoint else "⚠️"
        print(f"  {status} {checkpoint}: {count}/{expected_per_checkpoint} files")
    
    # Speakers breakdown
    print(f"\n🎤 SPEAKERS ({len(analysis['speakers'])})")
    expected_per_speaker = len(analysis['checkpoints']) * 5  # N checkpoints × 5 sentences
    for speaker in sorted(analysis['speakers'].keys()):
        count = analysis['speakers'][speaker]
        status = "✅" if count == expected_per_speaker else "⚠️"
        print(f"  {status} {speaker}: {count}/{expected_per_speaker} files")
    
    # Sentences breakdown
    print(f"\n📝 SENTENCES ({len(analysis['sentences'])})")
    expected_per_sentence = len(analysis['checkpoints']) * len(analysis['speakers'])
    for sentence_idx in sorted(analysis['sentences'].keys()):
        count = analysis['sentences'][sentence_idx]
        status = "✅" if count == expected_per_sentence else "⚠️"
        sentence_preview = HINDI_SENTENCES[sentence_idx-1][:50] + "..." if sentence_idx <= len(HINDI_SENTENCES) else "Unknown"
        print(f"  {status} Sentence {sentence_idx}: {count}/{expected_per_sentence} files")
        print(f"      \"{sentence_preview}\"")
    
    # Issues
    if analysis['missing_metadata']:
        print(f"\n⚠️  FILES WITH MISSING METADATA ({len(analysis['missing_metadata'])})")
        for filepath in analysis['missing_metadata'][:5]:  # Show first 5
            print(f"  - {os.path.basename(filepath)}")
        if len(analysis['missing_metadata']) > 5:
            print(f"  ... and {len(analysis['missing_metadata']) - 5} more")
    
    # Expected vs actual
    total_expected = len(analysis['checkpoints']) * len(analysis['speakers']) * 5
    print(f"\n📈 COMPLETENESS")
    print(f"Expected files: {total_expected}")
    print(f"Valid files: {analysis['valid_files']}")
    print(f"Completeness: {(analysis['valid_files']/total_expected)*100:.1f}%")
    
    return analysis['valid_files'] > 0

def show_sample_dataset_structure(analysis):
    """Show what the dataset structure will look like."""
    print(f"\n🏗️  DATASET STRUCTURE PREVIEW")
    print("=" * 60)
    
    print("The dataset will have these columns:")
    columns = [
        "audio", "filename", "checkpoint", "checkpoint_step", "speaker",
        "sentence_idx", "sentence_text", "sentence_category", "duration",
        "sample_rate", "channels", "file_size_mb", "language"
    ]
    
    for col in columns:
        print(f"  - {col}")
    
    print(f"\nSample rows (first 3 valid files):")
    valid_files = [f for f in analysis['files'] if f['valid']][:3]
    
    for i, file_info in enumerate(valid_files, 1):
        print(f"\n  Row {i}:")
        print(f"    filename: {file_info['filename']}")
        print(f"    checkpoint: {file_info['checkpoint']}")
        print(f"    speaker: {file_info['speaker']}")
        print(f"    sentence_idx: {file_info['sentence_idx']}")
        if file_info['sentence_idx'] <= len(HINDI_SENTENCES):
            text_preview = HINDI_SENTENCES[file_info['sentence_idx']-1][:60] + "..."
            print(f"    sentence_text: \"{text_preview}\"")
        print(f"    file_size_mb: {file_info['size_mb']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Preview batch inference results before uploading to Hugging Face")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing batch inference results"
    )
    parser.add_argument(
        "--save_report",
        type=str,
        help="Save detailed report to JSON file"
    )
    
    args = parser.parse_args()
    
    print("🔍 DATASET PREVIEW TOOL")
    print("=" * 60)
    print("This tool analyzes your batch inference results and shows")
    print("what will be uploaded to Hugging Face as a dataset.")
    print("=" * 60)
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Directory not found: {args.input_dir}")
        return 1
    
    # Analyze directory
    analysis = analyze_directory(args.input_dir)
    
    if not analysis:
        return 1
    
    # Print detailed report
    is_ready = print_analysis_report(analysis)
    
    # Show dataset structure preview
    show_sample_dataset_structure(analysis)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Detailed report saved to: {args.save_report}")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION")
    print("=" * 60)
    
    if is_ready and analysis['valid_files'] > 0:
        print("✅ Dataset is ready for upload!")
        print("You can proceed with the upload using:")
        print(f"  ./upload_dataset_runner.sh -i {args.input_dir} -r your-username/dataset-name")
        
        if analysis['missing_metadata']:
            print(f"\n⚠️  Note: {len(analysis['missing_metadata'])} files will be skipped due to missing metadata")
    else:
        print("❌ Dataset has issues that should be resolved before upload")
        print("Please check the files and directory structure")
    
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())