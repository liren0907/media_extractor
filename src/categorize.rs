use std::collections::{HashSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use image::DynamicImage;

use crate::config::Config;
use crate::image_processing::{
    calculate_hash_with_params, 
    calculate_enhanced_similarity
};

/// Categorize images based on similarity
pub fn categorize_images(config: &Config) -> Result<Option<BenchmarkResult>, Box<dyn std::error::Error>> {
    let source_dir = config.get_source_dir();
    let output_base_dir = config.get_output_dir();
    let start_time = std::time::Instant::now();

    println!("Analyzing images in: {}", source_dir.display());
    println!("Output directory: {}", output_base_dir.display());
    println!("Similarity threshold: {:.2}%", config.similarity_threshold * 100.0);
    println!("Using matching method: {:?}", config.matching_method);

    // Create output directory if it doesn't exist
    fs::create_dir_all(&output_base_dir)?;

    // Get all files in the directory (for counting total before filtering)
    let all_images: Vec<_> = fs::read_dir(&source_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && is_supported_image_file(&path, &config)
        })
        .map(|entry| entry.path())
        .collect();
    
    // Get all image files with filter applied
    let entries: Vec<_> = fs::read_dir(&source_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && is_supported_image_file(&path, &config) && 
            // Only include horizontal images if the filter is enabled
            (!config.filter_horizontal_only || is_horizontal_image(&path))
        })
        .map(|entry| entry.path())
        .collect();

    let total_images = entries.len();
    
    if entries.is_empty() {
        println!("No images found in directory: {}", source_dir.display());
        return Ok(None);
    }
    
    println!("Found {} images (of {} total) to process", entries.len(), all_images.len());
    if config.filter_horizontal_only {
        println!("{} images filtered out (vertical images)", all_images.len() - entries.len());
    }

    // Group similar images
    let mut categories: Vec<Vec<PathBuf>> = Vec::new();
    let mut processed_indices = HashSet::new();
    
    // Process images sequentially
    for i in 0..entries.len() {
        if processed_indices.contains(&i) {
            continue;
        }
        
        let path = &entries[i];
        processed_indices.insert(i);
        
        // Start a new category with this image
        let mut category = vec![path.clone()];
        
        // Compare with other images
        for j in 0..entries.len() {
            if i == j || processed_indices.contains(&j) {
                continue;
            }
            
            let other_path = &entries[j];
            
            // Use our enhanced similarity function
            let similarity = compare_images_enhanced(path, other_path, config);
            
            // Log all comparisons, not just matches
            println!("Comparison: {} vs {} = {:.2}%", 
                path.file_name().unwrap_or_default().to_string_lossy(),
                other_path.file_name().unwrap_or_default().to_string_lossy(),
                similarity * 100.0);
            
            // Add to category if similar enough
            if similarity >= config.similarity_threshold {
                category.push(other_path.clone());
                processed_indices.insert(j);
                
                println!("Match: {} similar to {} ({:.2}%)", 
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    other_path.file_name().unwrap_or_default().to_string_lossy(),
                    similarity * 100.0);
            }
        }
        
        // Add the category to our list
        categories.push(category);
    }
    
    // Process categories
    process_categories(&categories, config)?;
    
    // Calculate processing time
    let processing_time = start_time.elapsed();
    
    println!("Categorization completed. Found {} categories.", categories.len());
    println!("Processing time: {:?}", processing_time);
    
    // Create and return benchmark result if needed
    let benchmark_result = if config.benchmark_mode {
        let method_name = format!("{:?}", config.matching_method);
        Some(BenchmarkResult::new(method_name, processing_time, &categories, total_images))
    } else {
        None
    };
    
    Ok(benchmark_result)
}

// Helper function to compare two images with enhanced noise resistance
fn compare_images_enhanced(path1: &Path, path2: &Path, config: &Config) -> f64 {
    // Load the two images
    let img1 = match image::open(path1) {
        Ok(img) => img,
        Err(_) => return 0.0,
    };
    
    let img2 = match image::open(path2) {
        Ok(img) => img,
        Err(_) => return 0.0,
    };
    
    // Use our enhanced similarity calculation
    calculate_enhanced_similarity(&img1, &img2, config)
}

// Create a thumbnail for a category
fn create_category_thumbnail(
    representative_image: &Path, 
    category_name: &str, 
    config: &Config
) -> Result<(), Box<dyn std::error::Error>> {
    let thumbnail_dir = config.get_thumbnails_dir();
    fs::create_dir_all(&thumbnail_dir)?;
    
    let thumbnail_path = thumbnail_dir.join(format!("{}.jpg", category_name));
    let img = image::open(representative_image)?;
    
    // Create a nice sized thumbnail
    let thumbnail = img.resize(200, 200, image::imageops::FilterType::Lanczos3);
    thumbnail.save(&thumbnail_path)?;
    
    Ok(())
}

// Check if a file is a supported image format
fn is_supported_image_file(path: &Path, config: &Config) -> bool {
    match path.extension() {
        Some(ext) => {
            let ext_str = ext.to_string_lossy().to_lowercase();
            config.supported_formats.iter().any(|format| *format == ext_str)
        },
        None => false,
    }
}

// Check if an image is horizontal (width > height)
fn is_horizontal_image(path: &Path) -> bool {
    match image::image_dimensions(path) {
        Ok((width, height)) => width > height,
        Err(_) => false,
    }
}

/// Result data structure for benchmark mode
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub method_name: String,
    pub processing_time: std::time::Duration,
    pub categories_count: usize,
    pub total_images: usize,
    pub avg_category_size: f64,
}

impl BenchmarkResult {
    /// Creates a new benchmark result with the given parameters
    pub fn new(method_name: String, processing_time: std::time::Duration, 
               categories: &Vec<Vec<PathBuf>>, total_images: usize) -> Self {
        let categories_count = categories.len();
        let avg_category_size = if categories_count > 0 {
            categories.iter().map(|c| c.len()).sum::<usize>() as f64 / categories_count as f64
        } else {
            0.0
        };
        
        BenchmarkResult {
            method_name,
            processing_time,
            categories_count,
            total_images,
            avg_category_size,
        }
    }
    
    /// Formats the benchmark result as a string
    pub fn to_string(&self) -> String {
        format!(
            "Method: {}\n  Processing time: {:?}\n  Categories found: {}\n  Total images processed: {}\n  Average category size: {:.2}\n",
            self.method_name,
            self.processing_time,
            self.categories_count,
            self.total_images,
            self.avg_category_size
        )
    }
}

/// Process and save categories to their respective directories
fn process_categories(categories: &Vec<Vec<PathBuf>>, config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let output_base_dir = config.get_output_dir();
    let filtered_categories: Vec<_> = categories.iter()
        .filter(|category| category.len() > 1)  // Only include categories with at least 2 images
        .collect();
    
    println!("Grouped images into {} categories ({} with multiple images)", 
             categories.len(), filtered_categories.len());
    
    // Create category directories and copy images
    for (i, category) in filtered_categories.iter().enumerate() {
        let category_name = config.naming_pattern.replace("{index}", &format!("{:03}", i + 1));
        let category_dir = output_base_dir.join(&category_name);
        fs::create_dir_all(&category_dir)?;
        
        println!("Category {}: {} images", category_name, category.len());
        
        // Copy images to category directory
        let max_images = if config.max_images_per_category > 0 { 
            config.max_images_per_category 
        } else { 
            category.len() 
        };
        
        for (j, img_path) in category.iter().take(max_images).enumerate() {
            let filename = img_path.file_name()
                .unwrap_or_default()
                .to_string_lossy();
            
            let dest_path = category_dir.join(format!("{:03}_{}", j + 1, filename));
            fs::copy(img_path, &dest_path)?;
        }
        
        // Create category thumbnail if enabled
        if config.create_category_thumbnails && !category.is_empty() {
            create_category_thumbnail(&category[0], &category_name, config)?;
        }
    }
    
    println!("Category processing complete!");
    Ok(())
}
