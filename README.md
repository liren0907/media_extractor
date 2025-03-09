# Media-Extractor

A Rust application for extracting valuable and high-quality images from large collections of snapshots or similar images. Media-Extractor helps you identify the best representations while eliminating redundancy in your media library.

> **Note:** While designed with video frame extraction in mind, direct video file processing is not yet implemented in the current version.

## Quick Start

Get up and running with Media-Extractor in minutes:

```bash
# 1. Clone and build the project
git clone <repository-url>
cd media-extractor
cargo build --release

# 2. Create a test directory with some sample images
mkdir -p data/test
# Copy some test images to data/test

# 3. Run the application with default settings
cargo run --release

# 4. Check the results in the output directory
ls output_categorized
```

For first-time users, the natural high resolution split mode is recommended:

```bash
# Edit config.json to set run_mode to "natural_high_resolution_split"
# Then run:
cargo run --release
```

## Features

- **Enhanced Image Matching**: Combines multiple techniques (perceptual hash, color hash, SSIM) with weighted scoring
- **Noise Resistant Processing**: Applies blur preprocessing and supports bit error tolerance for better handling of noisy images
- **Structural Similarity (SSIM)**: Perceptual metric that aligns with human visual perception
- **Flexible Categorization**: Group images based on visual similarity with configurable thresholds
- **Smart Filtering**: Optional horizontal-only image filtering
- **Multiple Comparison Methods**: Support for various methods including perceptual hash, color hash, SIFT (with OpenCV), and SSIM
- **User-Friendly CLI**: Simple commands for categorization, comparison, and configuration management
- **Complete Configuration System**: Easily customize behavior through JSON configuration
- **Multiple Run Modes**: Support for different processing modes configurable via config file
- **Multi-threaded Processing**: Parallel processing for improved performance (configurable thread count)
- **Natural High Resolution Extraction**: Automatically extract highest resolution images from timestamp groups
- **Advanced Clustering**: Identify and extract distinct objects from the same timestamp group

## Installation

Ensure you have Rust and Cargo installed on your system. Then clone this repository (if you haven't already) and build the project:

```bash
git clone <repository-url>
cd media-extractor
cargo build --release
```

## Usage

### Basic Usage

For default operation using settings from `config.json`:

```bash
cargo run --release
```

This will run the mode specified by the `run_mode` field in your configuration file.

### Available Run Modes

Media-Extractor supports multiple processing modes:

1. **categorize**: Group similar images together based on visual similarity
2. **benchmark**: Test performance of different matching methods
3. **confusion_matrix**: Generate similarity matrix between images
4. **natural_high_resolution**: Extract highest resolution images by timestamp
5. **natural_high_resolution_split**: Extract highest resolution images of distinct objects by timestamp
6. **compare**: Compare similarity between two specific images

### Running Specific Modes

You can run a specific mode by providing it as a command-line argument:

```bash
# Basic categorization
cargo run categorize [/path/to/source/directory] [/path/to/output/directory]

# Natural high resolution extraction
cargo run natural_high_resolution <directory_path>

# Natural high resolution split (with clustering)
cargo run natural_high_resolution_split <directory_path>

# Compare specific images
cargo run compare <img1> <img2>
```

### Configuration Management

Media-Extractor uses a configuration file (`config.json`) to control its behavior:

```bash
# Create default configuration file
cargo run config create

# Show current configuration
cargo run config show
```

## How Image Matching Works

### Enhanced Similarity Algorithm

Media-Extractor uses a sophisticated multi-technique approach to image matching:

1. **Perceptual Hash Comparison** (40% weight): Creates a "fingerprint" for each image by:
   - Resizing the image to a configurable hash size (default: 16x16)
   - Converting to grayscale
   - Applying blur to reduce noise (configurable radius)
   - Comparing each pixel to the average to generate a binary hash
   - Calculating the Hamming distance between hashes

2. **Color Hash Comparison** (30% weight): Similar to perceptual hash, but preserves color information for more accurate matching.

3. **Structural Similarity (SSIM)** (30% weight): Evaluates the structural information, luminance, and contrast differences between images for a perceptually accurate comparison.

4. **Bit Error Tolerance**: Allows for more flexibility in matching to handle noisy images.

5. **Near-Threshold Boosting**: Implements special handling for borderline matches to improve categorization.

### Specialized Processing Modes

#### Natural High Resolution

The `natural_high_resolution` mode:
1. Groups images by timestamp (extracted from filenames)
2. For each timestamp group, selects the image with highest resolution
3. Copies selected images to the output directory

#### Natural High Resolution Split

The `natural_high_resolution_split` mode adds intelligence to the selection process:
1. Groups images by timestamp (extracted from filenames)
2. For each timestamp group:
   - Performs similarity-based clustering to identify distinct objects/scenes
   - For each cluster, selects the image with highest resolution
3. Copies all selected images to the output directory
4. Supports multi-threaded processing for improved performance

This mode is particularly useful when multiple distinct objects are photographed at the same timestamp.

### Categorization Process

The categorization algorithm:

1. Scans the specified input directory for supported image formats (jpg, png, gif, bmp, webp)
2. Applies optional filtering (e.g., horizontal-only images)
3. For each unprocessed image:
   - Compares it against all other unprocessed images
   - Groups images with similarity above the threshold (default: 90%)
   - Creates a directory for each category
   - Copies similar images to their corresponding category directory
   - Optionally generates thumbnails

## Configuration Options

The `config.json` file is central to Media-Extractor's functionality, allowing you to fine-tune every aspect of the image processing and categorization process. Here's a comprehensive breakdown of all available options:

```json
{
  "input_directory": "data/test",
  "output_directory": "data/categorized",
  "hash_size": 16,
  "similarity_threshold": 0.9,
  "consider_rotations": true,
  "consider_flips": true,
  "robust_comparison": true,
  "use_color_hash": true,
  "matching_method": "ssim",
  "blur_radius": 6.0,
  "bit_error_tolerance": 0.45,
  "sift_match_threshold": 0.35,
  "sift_feature_count": 800,
  "create_category_thumbnails": true,
  "max_images_per_category": 1000,
  "naming_pattern": "category_{index}",
  "log_level": "info",
  "filter_horizontal_only": true,
  "run_mode": "natural_high_resolution_split",
  "mode_options": {
    "benchmark": {
      "enabled": false
    },
    "confusion_matrix": {
      "source_directory": "data/test"
    },
    "natural_high_resolution": {
      "source_directory": "data/source"
    },
    "natural_high_resolution_split": {
      "source_directory": "data/source",
      "split_similarity_threshold": 0.85,
      "use_multi_threading": true,
      "thread_count": 4
    },
    "compare": {
      "image1": "data/test/segment_20250221_125843_1.jpg",
      "image2": "data/test/segment_20250221_125843_2.jpg"
    }
  }
}
```

### Configuration Parameters Explained

#### Required vs. Optional Parameters

Most configuration parameters have sensible defaults and are optional. The only required parameters are:
- `input_directory`
- `output_directory`

All other parameters will use their default values if omitted from the config file.

#### Directory Settings
- **input_directory**: Path to the source directory containing images to process. Can be relative or absolute.
  - Default: `"data/frames"`
  - Impact: Determines where Media-Extractor looks for images to categorize.

- **output_directory**: Path where categorized images and generated data will be stored.
  - Default: `"data/categorized"`
  - Impact: Defines the location for output files, including category folders and thumbnails.

#### Image Processing Settings
- **hash_size**: Size of the perceptual hash in pixels (NxN). Larger values capture more image detail.
  - Default: `8` (8x8 pixels)
  - Current value: `16` (16x16 pixels)
  - Impact: Higher values increase accuracy but reduce tolerance to minor variations and increase processing time.

- **similarity_threshold**: Minimum similarity score (0.0-1.0) required to consider two images as matching.
  - Default: `0.85` (85%)
  - Current value: `0.9` (90%)
  - Impact: Higher values create more categories with fewer images per category, requiring more similarity for grouping.

- **supported_formats**: *(Optional - has default values)*
  - Default: `["jpg", "jpeg", "png", "gif", "bmp", "webp"]`
  - Impact: Determines which file types are included in the processing.
  - Note: This field is now optional in the config file and will use the default values if omitted.

#### Similarity Comparison Settings
- **consider_rotations**: Whether to check for similarities in rotated versions of images.
  - Default: `true`
  - Impact: When enabled, detects similar images even if they're rotated, at the cost of additional processing time.

- **consider_flips**: Whether to check for similarities in horizontally/vertically flipped images.
  - Default: `true`
  - Impact: When enabled, detects similar images even if they're mirrored, at the cost of additional processing time.

- **robust_comparison**: Whether to use higher quality image resizing for more accurate results.
  - Default: `true`
  - Impact: When enabled, uses Lanczos3 filter instead of Triangle filter for more accurate but slower image resizing.

- **use_color_hash**: Whether to incorporate color information in the perceptual hash.
  - Default: `false`
  - Current value: `true`
  - Impact: When enabled, compares colors in addition to patterns, making the comparison more accurate for color-sensitive content.

- **matching_method**: Algorithm used for image comparison.
  - Options: `"perceptualhash"`, `"colorhash"`, `"sift"`, `"surf"`, `"ssim"`
  - Default: `"perceptualhash"`
  - Current value: `"ssim"` (Structural Similarity Index)
  - Impact: Different methods offer varying trade-offs between accuracy, speed, and noise tolerance.

#### Noise Handling Settings
- **blur_radius**: Strength of Gaussian blur applied to images before comparison.
  - Default: `1.5`
  - Current value: `6.0`
  - Impact: Higher values reduce noise but might obscure fine details. Values around 5-7 work well for noisy images.

- **bit_error_tolerance**: Percentage of hash bits that can differ while still considering images as similar.
  - Default: `0.1` (10%)
  - Current value: `0.45` (45%)
  - Impact: Higher values are more tolerant of noise and variations, but may group dissimilar images.

#### SIFT/SURF Specific Settings (for OpenCV Feature Detection)
- **sift_match_threshold**: Threshold for SIFT/SURF feature matching.
  - Default: `0.6` (60%)
  - Current value: `0.35` (35%)
  - Impact: Lower values are more tolerant of perspective changes and partial matches.

- **sift_feature_count**: Maximum number of features to extract with SIFT/SURF.
  - Default: `0` (use OpenCV default)
  - Current value: `800`
  - Impact: Higher values capture more details but increase processing time.

#### Categorization Settings
- **create_category_thumbnails**: Whether to generate thumbnail images for each category.
  - Default: `false`
  - Current value: `true`
  - Impact: When enabled, creates a thumbnail representation of each category for easier browsing.

- **max_images_per_category**: Maximum number of images to include in a single category.
  - Default: `1000`
  - Impact: Prevents categories from becoming too large, which is useful for large datasets.

- **naming_pattern**: Pattern used for naming category folders.
  - Default: `"category_{index}"`
  - Impact: Determines how category folders are named. `{index}` is replaced with a sequential number.

#### General and Filtering Settings
- **log_level**: Controls the verbosity of application logging.
  - Options: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`
  - Default: `"info"`
  - Impact: Higher verbosity levels (`debug`, `trace`) show more detailed processing information.

- **filter_horizontal_only**: Whether to process only horizontally-oriented images.
  - Default: `false`
  - Current value: `true`
  - Impact: When enabled, vertical/portrait images are skipped during processing.

#### Run Mode Settings
- **run_mode**: Specifies the default processing mode when no command-line argument is provided.
  - Options: `"categorize"`, `"benchmark"`, `"confusion_matrix"`, `"natural_high_resolution"`, `"natural_high_resolution_split"`, `"compare"`
  - Default: `"categorize"`
  - Impact: Determines which processing mode to run when no command is provided.
  - Note: This replaces the older `benchmark_mode` flag which is now deprecated.

- **mode_options**: Contains mode-specific configuration settings.
  - **benchmark**: Options for benchmark mode.
    - `enabled`: Whether benchmark mode is enabled.
  - **confusion_matrix**: Options for confusion matrix mode.
    - `source_directory`: Directory containing images for confusion matrix.
  - **natural_high_resolution**: Options for natural high resolution mode.
    - `source_directory`: Directory containing source images.
  - **natural_high_resolution_split**: Options for natural high resolution split mode.
    - `source_directory`: Directory containing source images.
    - `split_similarity_threshold`: Threshold for determining similarity clusters.
    - `use_multi_threading`: Whether to use multi-threaded processing.
    - `thread_count`: Number of threads to use when multi-threading is enabled.
  - **compare**: Options for compare mode.
    - `image1`, `image2`: Paths to images to compare.

### Optimizing Configuration for Your Needs

- **For noisy or low-quality images**: Increase `blur_radius` (4-8) and `bit_error_tolerance` (0.3-0.5)
- **For high-precision matching**: Increase `hash_size` (16-32) and use `"ssim"` as the matching method
- **For faster processing**: Reduce `hash_size` (4-8) and disable `robust_comparison`
- **For color-sensitive matching**: Enable `use_color_hash` and consider using higher `similarity_threshold`
- **For matching despite perspective changes**: Use SIFT with lower `sift_match_threshold` values
- **For better performance in natural_high_resolution_split**: Enable `use_multi_threading` and set `thread_count` to match your CPU core count

### Sample Configurations for Different Run Modes

Below are minimal example configurations for each run mode. You can use these as starting points and customize as needed.

#### Categorize Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/categorized",
  "similarity_threshold": 0.9,
  "run_mode": "categorize"
}
```

#### Benchmark Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/benchmarks",
  "run_mode": "benchmark",
  "mode_options": {
    "benchmark": {
      "enabled": true
    }
  }
}
```

#### Confusion Matrix Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/output",
  "run_mode": "confusion_matrix",
  "mode_options": {
    "confusion_matrix": {
      "source_directory": "data/test_samples"
    }
  }
}
```

#### Natural High Resolution Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/output",
  "run_mode": "natural_high_resolution",
  "mode_options": {
    "natural_high_resolution": {
      "source_directory": "data/originals"
    }
  }
}
```

#### Natural High Resolution Split Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/output",
  "run_mode": "natural_high_resolution_split",
  "mode_options": {
    "natural_high_resolution_split": {
      "source_directory": "data/originals",
      "split_similarity_threshold": 0.85,
      "use_multi_threading": true,
      "thread_count": 4
    }
  }
}
```

#### Compare Mode
```json
{
  "input_directory": "data/frames",
  "output_directory": "data/output",
  "run_mode": "compare",
  "mode_options": {
    "compare": {
      "image1": "data/test/image1.jpg",
      "image2": "data/test/image2.jpg"
    }
  }
}
```

## Advanced Usage

### Tuning Natural High Resolution Split Mode

The `natural_high_resolution_split` mode offers several tuning options:

- **split_similarity_threshold**: Controls how images are clustered.
  - Higher values (0.9+): More strict clustering, creates more clusters with fewer images each.
  - Lower values (0.7-0.8): More lenient clustering, creates fewer clusters with more images each.
  - Default: `0.85` (balanced approach)

- **Multi-threading settings**:
  - `use_multi_threading`: Enable/disable parallel processing.
  - `thread_count`: Number of processing threads (set to CPU core count for optimal performance).

### Horizontal Filtering

The `filter_horizontal_only` setting affects all modes including `natural_high_resolution_split`:

- When enabled, only processes images where width > height
- Speeds up processing by ignoring vertical/portrait images
- Applied during initial image loading to maximize efficiency

## Performance Considerations

- The image comparison has O(n²) complexity where n is the number of images
- Use `--release` flag for significantly better performance
- Larger hash sizes provide more accuracy but slower processing
- SIFT-based matching (when OpenCV is enabled) is more compute-intensive but can be more accurate for certain images

## License

MIT
