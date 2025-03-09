use media_extractor::config::{Config, load_config, RunMode};
use media_extractor::categorize;
use media_extractor::image_processing;
use media_extractor::benchmark;
use media_extractor::confusion_matrix;
use media_extractor::natural_categorization;
use media_extractor::natural_categorization_split;
use std::path::Path;
use std::env;

// Main application
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();
    
    // If no arguments provided, run the default mode from config
    if args.len() <= 1 {
        return run_from_config();
    }
    
    // Handle different commands
    match args[1].as_str() {
        "categorize" => {
            // Load configuration
            let mut config = load_config()?;
            
            // Override input/output directories if provided
            if args.len() > 2 {
                config.input_directory = args[2].clone();
            }
            if args.len() > 3 {
                config.output_directory = args[3].clone();
            }
            
            // Run categorization
            categorize::categorize_images(&config)?;
        },
        "benchmark" => {
            // Load configuration
            let mut config = load_config()?;
            
            // Override input/output directories if provided
            if args.len() > 2 {
                config.input_directory = args[2].clone();
            }
            if args.len() > 3 {
                config.output_directory = args[3].clone();
            }
            
            // Run benchmark
            benchmark::run_benchmark(&config)?;
        },
        "confusion_matrix" => {
            if args.len() < 3 {
                println!("Usage: media-lake confusion_matrix <directory_path>");
                println!("  Generates a confusion matrix of image similarities in the specified directory");
                return Ok(());
            }
            
            // Load configuration
            let config = load_config()?;
            
            // Get source directory from args
            let source_dir = &args[2];
            
            // Run confusion matrix generation
            confusion_matrix::run_confusion_matrix(&config, source_dir)?;
        },
        "natural_high_resolution" => {
            if args.len() < 3 {
                println!("Usage: media-lake natural_high_resolution <directory_path>");
                println!("  Extracts highest resolution images from each natural timestamp group");
                return Ok(());
            }
            
            // Load configuration
            let config = load_config()?;
            
            // Get source directory from args
            let source_dir = &args[2];
            
            // Run natural high resolution mode
            natural_categorization::run_natural_high_resolution(&config, source_dir)?;
        },
        "natural_high_resolution_split" => {
            if args.len() < 3 {
                println!("Usage: media-lake natural_high_resolution_split <directory_path>");
                println!("  Extracts highest resolution images by timestamp with clustering");
                return Ok(());
            }
            
            // Load configuration
            let config = load_config()?;
            
            // Get source directory from args
            let source_dir = &args[2];
            
            // Run natural high resolution split mode
            natural_categorization_split::run_natural_high_resolution_split(&config, source_dir)?;
        },
        "compare" => {
            if args.len() < 4 {
                println!("Usage: media-lake compare <image1> <image2>");
                return Ok(());
            }
            
            // Load the images
            let img1_path = &args[2];
            let img2_path = &args[3];
            
            // Load configuration for comparison settings
            let config = load_config()?;
            
            // Compare the images
            compare_images(img1_path, img2_path, &config)?;
        },
        "config" => {
            if args.len() < 3 {
                println!("Usage: media-lake config <command> [args]");
                println!("Commands:");
                println!("  create - Create default configuration file");
                println!("  show   - Show current configuration");
                println!("  check  - Check configuration settings");
                println!("  set <key> <value> - Modify a configuration setting");
                return Ok(());
            }
            
            match args[2].as_str() {
                "create" => create_config()?,
                "show" => show_config()?,
                "check" => check_config()?,
                "set" => {
                    if args.len() < 5 {
                        println!("Usage: media-lake config set <key> <value>");
                        return Ok(());
                    }
                    set_config(&args[3], &args[4])?;
                },
                _ => {
                    println!("Unknown config command: {}", args[2]);
                }
            }
        },
        "help" => {
            print_help();
        },
        _ => {
            println!("Unknown command: {}", args[1]);
            print_help();
        }
    }
    
    Ok(())
}

/// Run the appropriate mode based on configuration
fn run_from_config() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = load_config()?;
    
    println!("Running mode from configuration: {:?}", config.run_mode);
    
    // Check which mode to run
    match config.run_mode {
        RunMode::Categorize => {
            run_default_categorization()
        },
        RunMode::Benchmark => {
            // Benchmark mode is directly enabled by RunMode::Benchmark
            benchmark::run_benchmark(&config)
        },
        RunMode::ConfusionMatrix => {
            let source_dir = &config.mode_options.confusion_matrix.source_directory;
            println!("Running confusion matrix on directory: {}", source_dir);
            confusion_matrix::run_confusion_matrix(&config, source_dir)
        },
        RunMode::NaturalHighResolution => {
            let source_dir = &config.mode_options.natural_high_resolution.source_directory;
            println!("Running natural high resolution on directory: {}", source_dir);
            natural_categorization::run_natural_high_resolution(&config, source_dir)
        },
        RunMode::NaturalHighResolutionSplit => {
            let source_dir = &config.mode_options.natural_high_resolution_split.source_directory;
            println!("Running natural high resolution split on directory: {}", source_dir);
            natural_categorization_split::run_natural_high_resolution_split(&config, source_dir)
        },
        RunMode::Compare => {
            let img1_path = &config.mode_options.compare.image1;
            let img2_path = &config.mode_options.compare.image2;
            println!("Comparing images: {} and {}", img1_path, img2_path);
            compare_images(img1_path, img2_path, &config)
        }
    }
}

/// Legacy method to maintain backward compatibility
fn run_default_mode() -> Result<(), Box<dyn std::error::Error>> {
    // Redirect to the new method
    run_from_config()
}

fn run_default_categorization() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = load_config()?;
    println!("Loaded configuration");

    // Create required directories if they don't exist
    create_directories(&config)?;
    
    // Print information about the horizontal filter
    if config.filter_horizontal_only {
        println!("Horizontal image filter is ENABLED - only processing images where width > height");
    } else {
        println!("Horizontal image filter is DISABLED - processing all images regardless of dimensions");
    }
    
    // Print information about the color hash
    if config.use_color_hash {
        println!("Color-aware image comparison is ENABLED - comparing colors across RGB channels");
    } else {
        println!("Grayscale image comparison is ENABLED - color information is ignored");
    }
    
    // Print noise handling settings
    println!("Noise handling: Blur radius = {:.1}, Bit error tolerance = {:.1}%", 
             config.blur_radius, config.bit_error_tolerance * 100.0);
    println!("Similarity threshold: {:.1}%", config.similarity_threshold * 100.0);

    // Run categorization
    categorize::categorize_images(&config)?;

    Ok(())
}

/// Create all necessary directories based on configuration
fn create_directories(config: &Config) -> Result<(), std::io::Error> {
    // Get all required directories
    let dirs = [
        config.get_source_dir(),
        config.get_data_dir(),
        config.get_output_dir(),
        config.get_thumbnails_dir()
    ];
    
    // Create them if they don't exist
    for dir in dirs.iter() {
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
        }
    }
    
    Ok(())
}

/// Compare two images and print similarity score
fn compare_images(img1_path: &str, img2_path: &str, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing images:");
    println!("  1: {}", img1_path);
    println!("  2: {}", img2_path);
    
    // Load images
    let img1 = image::open(img1_path)?;
    let img2 = image::open(img2_path)?;
    
    // Calculate similarity
    let similarity = if config.robust_comparison {
        let hash1 = image_processing::calculate_hash_with_params(&img1, config.hash_size, config.blur_radius);
        let hash2 = image_processing::calculate_hash_with_params(&img2, config.hash_size, config.blur_radius);
        image_processing::calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
    } else {
        let hash1 = image_processing::calculate_hash(&img1, config);
        let hash2 = image_processing::calculate_hash(&img2, config);
        image_processing::calculate_similarity(&hash1, &hash2)
    };
    
    println!("Similarity: {:.2}%", similarity * 100.0);
    
    if similarity >= config.similarity_threshold {
        println!("Result: Images are SIMILAR (above threshold of {:.2}%)", config.similarity_threshold * 100.0);
    } else {
        println!("Result: Images are DIFFERENT (below threshold of {:.2}%)", config.similarity_threshold * 100.0);
    }
    
    Ok(())
}

/// Create default configuration file
fn create_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::default();
    config.save_to_file("config.json")?;
    println!("Created default configuration file: config.json");
    Ok(())
}

/// Show current configuration
fn show_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config()?;
    println!("{:#?}", config);
    Ok(())
}

/// Check configuration settings
fn check_config() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config()?;
    
    // Check base directories
    let source_dir = config.get_source_dir();
    let output_dir = config.get_output_dir();
    
    println!("Configuration check:");
    println!("  Input directory: {}", source_dir.display());
    println!("  Output directory: {}", output_dir.display());
    println!("  Similarity threshold: {:.2}%", config.similarity_threshold * 100.0);
    println!("  Hash size: {}", config.hash_size);
    
    Ok(())
}

/// Set configuration setting
fn set_config(key: &str, value: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = load_config()?;
    
    // Update configuration based on key
    match key {
        "similarity.threshold" => {
            let threshold = value.parse::<f64>()?;
            if threshold < 0.0 || threshold > 1.0 {
                return Err("Threshold must be between 0.0 and 1.0".into());
            }
            config.similarity_threshold = threshold;
        },
        "hash_size" => {
            let size = value.parse::<u32>()?;
            if size < 4 || size > 64 {
                return Err("Hash size must be between 4 and 64".into());
            }
            config.hash_size = size;
        },
        "input_directory" => {
            config.input_directory = value.to_string();
        },
        "output_directory" => {
            config.output_directory = value.to_string();
        },
        "use_color_hash" => {
            config.use_color_hash = value.parse::<bool>()?;
        },
        "consider_rotations" => {
            config.consider_rotations = value.parse::<bool>()?;
        },
        "consider_flips" => {
            config.consider_flips = value.parse::<bool>()?;
        },
        _ => {
            return Err(format!("Unknown configuration key: {}", key).into());
        }
    }
    
    // Save updated configuration
    config.save_to_file("config.json")?;
    println!("Updated configuration saved");
    Ok(())
}

/// Print help information
fn print_help() {
    println!("Media Lake - Image Categorization Tool");
    println!();
    println!("Commands:");
    println!("  categorize [input_dir] [output_dir]       - Categorize images based on similarity");
    println!("  benchmark [input_dir] [output_dir]        - Run benchmark tests with all matching methods");
    println!("  confusion_matrix <directory_path>         - Generate a confusion matrix for image similarities");
    println!("  natural_high_resolution <directory_path>  - Extract highest resolution images by timestamp");
    println!("  natural_high_resolution_split <directory_path> - Extract highest resolution images by timestamp with clustering");
    println!("  compare <image1> <image2>                 - Compare two images and show similarity score");
    println!("  config <subcommand>                       - Manage configuration");
    println!("  help                                      - Show this help message");
    println!();
    println!("Running without arguments:");
    println!("  The application will run according to the \"run_mode\" setting in config.json");
    println!("  All configuration for modes can be specified in the \"mode_options\" section");
    println!();
    println!("Available run_mode values:");
    println!("  - categorize");
    println!("  - benchmark");
    println!("  - confusion_matrix");
    println!("  - natural_high_resolution");
    println!("  - natural_high_resolution_split");
    println!("  - compare");
}
