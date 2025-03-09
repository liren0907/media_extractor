use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Matching method for image comparison
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MatchingMethod {
    /// Perceptual hash based matching (default)
    PerceptualHash,
    /// Color-aware perceptual hash
    ColorHash,
    /// SIFT feature detection and matching
    Sift,
    /// SURF feature detection and matching
    Surf,
    /// Structural similarity index
    Ssim
}

/// Available run modes for the application
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Default categorization mode
    Categorize,
    /// Benchmark testing for all methods
    Benchmark,
    /// Generate confusion matrix
    ConfusionMatrix,
    /// Extract high resolution images by timestamp
    NaturalHighResolution,
    /// Extract high resolution images by timestamp with split clustering
    NaturalHighResolutionSplit,
    /// Compare two specific images
    Compare
}

impl Default for RunMode {
    fn default() -> Self {
        RunMode::Categorize
    }
}

impl Default for MatchingMethod {
    fn default() -> Self {
        MatchingMethod::PerceptualHash
    }
}

/// Mode-specific options for Confusion Matrix
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConfusionMatrixOptions {
    #[serde(default = "default_confusion_matrix_source")]
    pub source_directory: String,
}

impl Default for ConfusionMatrixOptions {
    fn default() -> Self {
        ConfusionMatrixOptions {
            source_directory: default_confusion_matrix_source(),
        }
    }
}

/// Mode-specific options for Natural High Resolution
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NaturalHighResolutionOptions {
    #[serde(default = "default_natural_hr_source")]
    pub source_directory: String,
}

impl Default for NaturalHighResolutionOptions {
    fn default() -> Self {
        NaturalHighResolutionOptions {
            source_directory: default_natural_hr_source(),
        }
    }
}

/// Mode-specific options for Compare
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompareOptions {
    #[serde(default = "default_compare_image1")]
    pub image1: String,
    #[serde(default = "default_compare_image2")]
    pub image2: String,
}

impl Default for CompareOptions {
    fn default() -> Self {
        CompareOptions {
            image1: default_compare_image1(),
            image2: default_compare_image2(),
        }
    }
}

/// Mode-specific options for Natural High Resolution Split
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NaturalHighResolutionSplitOptions {
    #[serde(default = "default_natural_hr_source")]
    pub source_directory: String,
    #[serde(default = "default_split_similarity_threshold")]
    pub split_similarity_threshold: f64,
    #[serde(default)]
    pub use_multi_threading: bool,
    #[serde(default = "default_thread_count")]
    pub thread_count: usize,
}

impl Default for NaturalHighResolutionSplitOptions {
    fn default() -> Self {
        NaturalHighResolutionSplitOptions {
            source_directory: default_natural_hr_source(),
            split_similarity_threshold: default_split_similarity_threshold(),
            use_multi_threading: true,
            thread_count: default_thread_count(),
        }
    }
}

/// Holds all mode-specific configuration options
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModeOptions {
    #[serde(default)]
    pub benchmark: BenchmarkOptions,
    #[serde(default)]
    pub confusion_matrix: ConfusionMatrixOptions,
    #[serde(default)]
    pub natural_high_resolution: NaturalHighResolutionOptions,
    #[serde(default)]
    pub natural_high_resolution_split: NaturalHighResolutionSplitOptions,
    #[serde(default)]
    pub compare: CompareOptions,
}

impl Default for ModeOptions {
    fn default() -> Self {
        ModeOptions {
            benchmark: Default::default(),
            confusion_matrix: Default::default(),
            natural_high_resolution: Default::default(),
            natural_high_resolution_split: Default::default(),
            compare: Default::default(),
        }
    }
}

/// Options for benchmark mode
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkOptions {
    #[serde(default = "default_benchmark_methods")]
    pub methods: Vec<MatchingMethod>,
}

fn default_benchmark_methods() -> Vec<MatchingMethod> {
    vec![
        MatchingMethod::PerceptualHash,
        MatchingMethod::ColorHash,
        MatchingMethod::Ssim,
    ]
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        BenchmarkOptions {
            methods: default_benchmark_methods(),
        }
    }
}

/// Application configuration structure that matches config.json
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    // Directory paths
    pub input_directory: String,
    pub output_directory: String,
    
    // Image processing settings
    pub hash_size: u32,
    pub similarity_threshold: f64,
    #[serde(default = "default_supported_formats")]
    pub supported_formats: Vec<String>,
    
    // Similarity comparison settings
    pub consider_rotations: bool,
    pub consider_flips: bool,
    pub robust_comparison: bool,
    #[serde(default)]
    pub use_color_hash: bool,
    #[serde(default)]
    pub matching_method: MatchingMethod,
    
    // Noise handling settings
    #[serde(default = "default_blur_radius")]
    pub blur_radius: f32,
    #[serde(default = "default_bit_error_tolerance")]
    pub bit_error_tolerance: f64,
    
    // SIFT specific settings
    #[serde(default = "default_sift_match_threshold")]
    pub sift_match_threshold: f64,
    #[serde(default = "default_sift_feature_count")]
    pub sift_feature_count: i32,
    
    // Categorization settings
    pub create_category_thumbnails: bool,
    pub max_images_per_category: usize,
    pub naming_pattern: String,
    
    // General settings
    #[serde(default = "default_log_level")]
    pub log_level: String,
    
    // Image filter settings
    #[serde(default)]
    pub filter_horizontal_only: bool,
    
    // Benchmark mode settings (deprecated, use run_mode and mode_options instead)
    #[serde(default)]
    pub benchmark_mode: bool,
    
    // Mode selection and options
    #[serde(default)]
    pub run_mode: RunMode,
    
    #[serde(default)]
    pub mode_options: ModeOptions,
}

// Default functions for parameters
fn default_blur_radius() -> f32 {
    1.5
}

fn default_supported_formats() -> Vec<String> {
    vec![
        "jpg".to_string(),
        "jpeg".to_string(),
        "png".to_string(),
        "gif".to_string(),
        "bmp".to_string(),
        "webp".to_string(),
    ]
}

fn default_bit_error_tolerance() -> f64 {
    0.1 // 10% tolerance for bit errors in noisy images
}

fn default_sift_match_threshold() -> f64 {
    0.6 // 60% of keypoints should match to consider similar
}

fn default_sift_feature_count() -> i32 {
    0 // 0 means use OpenCV default
}

fn default_confusion_matrix_source() -> String {
    "data/test".to_string()
}

fn default_natural_hr_source() -> String {
    "data/test".to_string()
}

fn default_compare_image1() -> String {
    "data/test/image1.jpg".to_string()
}

fn default_compare_image2() -> String {
    "data/test/image2.jpg".to_string()
}

fn default_split_similarity_threshold() -> f64 {
    0.85 // 85% similarity threshold for splitting clusters
}

fn default_thread_count() -> usize {
    // Default to the number of logical cores, but at least 2
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(2)
}

fn default_log_level() -> String {
    "trace".to_string()
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }

    /// Get filter type based on configuration
    pub fn get_filter_type(&self) -> FilterType {
        if self.robust_comparison {
            FilterType::Lanczos3
        } else {
            FilterType::Triangle
        }
    }

    /// Get source directory
    pub fn get_source_dir(&self) -> PathBuf {
        PathBuf::from(&self.input_directory)
    }

    /// Get output directory
    pub fn get_output_dir(&self) -> PathBuf {
        PathBuf::from(&self.output_directory)
    }

    /// Get the absolute path for the source directory
    pub fn get_source_dir_absolute(&self) -> PathBuf {
        let source_dir = Path::new(&self.input_directory);
        if source_dir.is_absolute() {
            source_dir.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(source_dir)
        }
    }
    
    /// Get the absolute path for the output directory
    pub fn get_output_dir_absolute(&self) -> PathBuf {
        let output_dir = Path::new(&self.output_directory);
        if output_dir.is_absolute() {
            output_dir.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(output_dir)
        }
    }
    
    /// Get the base directory (parent of input directory)
    pub fn get_base_dir(&self) -> PathBuf {
        self.get_source_dir_absolute().parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
    }
    
    /// For backward compatibility - returns the source directory (same as get_source_dir)
    pub fn get_source_images_dir(&self) -> PathBuf {
        self.get_source_dir_absolute()
    }
    
    /// For backward compatibility - processed data will be stored in output_directory/processed
    pub fn get_data_dir(&self) -> PathBuf {
        self.get_output_dir_absolute().join("processed")
    }

    /// Check if a file extension is supported
    pub fn is_supported_format(&self, extension: &str) -> bool {
        self.supported_formats
            .iter()
            .any(|format| format.eq_ignore_ascii_case(extension))
    }
    
    /// Get the storage root directory
    pub fn get_storage_root(&self) -> PathBuf {
        self.get_output_dir_absolute()
    }

    /// Get the cached images directory
    pub fn get_cache_dir(&self) -> PathBuf {
        self.get_storage_root().join("cache")
    }

    /// Get the thumbnail images directory
    pub fn get_thumbnails_dir(&self) -> PathBuf {
        self.get_storage_root().join("thumbnails")
    }
 
    /// Get the path to the processed files list
    pub fn get_processed_list_path(&self) -> PathBuf {
        self.get_storage_root().join("processed_files.json")
    }

    /// Get public API endpoint (for external systems)
    pub fn get_public_api_endpoint(&self) -> String {
        "http://localhost:8080/api".to_string()
    }
    
    /// Get the path for thumbnails with specific dimensions
    pub fn get_sized_thumbnails_dir(&self, width: u32, height: u32) -> PathBuf {
        self.get_thumbnails_dir().join(format!("{}x{}", width, height))
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Get the default configuration
    pub fn default() -> Self {
        Self {
            // Directory paths
            input_directory: "data/frames".to_string(),
            output_directory: "data/categorized".to_string(),
            
            // Image processing settings
            hash_size: 8,
            similarity_threshold: 0.85,
            supported_formats: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "gif".to_string(),
                "bmp".to_string(),
                "webp".to_string(),
            ],
            
            // Similarity comparison settings
            consider_rotations: true,
            consider_flips: true,
            robust_comparison: true,
            use_color_hash: false,
            matching_method: MatchingMethod::PerceptualHash,
            
            // Noise handling settings
            blur_radius: 1.5,
            bit_error_tolerance: 0.1,
            
            // SIFT specific settings
            sift_match_threshold: 0.6,
            sift_feature_count: 0,
            
            // Categorization settings
            create_category_thumbnails: false,
            max_images_per_category: 1000,
            naming_pattern: "category_{index}".to_string(),
            
            // General settings
            log_level: "trace".to_string(),
            
            // Image filter settings
            filter_horizontal_only: false,
            
            // Benchmark mode settings
            benchmark_mode: false,
            
            // Mode selection and options
            run_mode: RunMode::Categorize,
            mode_options: ModeOptions::default(),
        }
    }
}

/// Load the configuration, creating a default one if it doesn't exist
pub fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let config_path = "config.json";
    
    if !std::path::Path::new(config_path).exists() {
        let default_config = Config::default();
        default_config.save_to_file(config_path)?;
        println!("Created default configuration file: {}", config_path);
    }
    
    let mut config = Config::from_file(config_path)?;

    // Validate and apply post-load logic
    if config.log_level.is_empty() {
        config.log_level = default_log_level();
    }

    Ok(config)
}
