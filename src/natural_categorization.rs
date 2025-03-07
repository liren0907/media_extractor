use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use regex::Regex;
use image::GenericImageView;

use crate::config::Config;

/// Run natural high resolution mode
pub fn run_natural_high_resolution(
    config: &Config,
    source_dir: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    println!("Running Natural High Resolution mode");
    println!("Source directory: {}", source_dir);
    
    // Validate source directory
    let source_path = Path::new(source_dir);
    if !source_path.exists() || !source_path.is_dir() {
        return Err(format!("Source directory does not exist or is not a directory: {}", source_dir).into());
    }
    
    // Create output directory
    let dir_name = source_path.file_name()
        .unwrap_or_default()
        .to_string_lossy();
    
    let output_base_dir = config.get_output_dir().join("natural_high_resolution").join(&*dir_name);
    fs::create_dir_all(&output_base_dir)?;
    
    // Get all image files
    let image_paths: Vec<PathBuf> = fs::read_dir(source_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && is_supported_image_file(&path, config)
        })
        .map(|entry| entry.path())
        .collect();
    
    if image_paths.is_empty() {
        return Err(format!("No supported images found in {}", source_dir).into());
    }
    
    println!("Found {} images to process", image_paths.len());
    
    // Categorize images by timestamp
    let categorized_images = categorize_by_timestamp(&image_paths)?;
    println!("Grouped into {} natural categories", categorized_images.len());
    
    // Find highest resolution image for each category
    let mut report_entries = Vec::new();
    let mut total_selected = 0;
    
    for (timestamp, images) in &categorized_images {
        println!("Processing group: {} ({} images)", timestamp, images.len());
        
        // Find highest resolution image
        let (best_image, width, height, size) = find_highest_resolution(images)?;
        
        // Get filename for the best image
        let filename = best_image.file_name()
            .unwrap_or_default()
            .to_string_lossy();
        
        // Output filename with timestamp prefix to avoid collisions
        let output_filename = format!("best_{}", filename);
        let output_path = output_base_dir.join(&output_filename);
        
        // Copy to output directory
        fs::copy(&best_image, &output_path)?;
        
        // Add to report
        report_entries.push(ReportEntry {
            timestamp: timestamp.clone(),
            image_count: images.len(),
            selected_image: filename.to_string(),
            width,
            height,
            file_size: size,
        });
        
        total_selected += 1;
        println!("Selected: {} ({}x{}, {:.2} MB)", 
                 filename, width, height, size as f64 / (1024.0 * 1024.0));
    }
    
    // Generate report
    let report_path = output_base_dir.join("selection_report.txt");
    generate_report(&report_entries, &report_path, start_time.elapsed())?;
    
    // Generate CSV report
    let csv_path = output_base_dir.join("selection_report.csv");
    generate_csv_report(&report_entries, &csv_path)?;
    
    println!("\nNatural High Resolution processing complete!");
    println!("Selected {} highest resolution images from {} groups", 
             total_selected, categorized_images.len());
    println!("Results saved to: {}", output_base_dir.display());
    println!("Processing time: {:?}", start_time.elapsed());
    
    Ok(())
}

/// Data structure for report entries
struct ReportEntry {
    timestamp: String,
    image_count: usize,
    selected_image: String,
    width: u32,
    height: u32,
    file_size: u64,
}

/// Categorize images by timestamp in filename
fn categorize_by_timestamp(
    images: &[PathBuf]
) -> Result<HashMap<String, Vec<PathBuf>>, Box<dyn std::error::Error>> {
    // Define regex pattern for filename extraction
    // Format: segment_YYYYMMDD_HHMMSS_N.jpg
    let re = Regex::new(r"segment_(\d{8})_(\d{6})_\d+\.jpe?g$")?;
    
    let mut categorized: HashMap<String, Vec<PathBuf>> = HashMap::new();
    
    for image_path in images {
        if let Some(filename) = image_path.file_name() {
            if let Some(filename_str) = filename.to_str() {
                if let Some(captures) = re.captures(filename_str) {
                    // Extract date and time components
                    let date = captures.get(1).map_or("", |m| m.as_str());
                    let time = captures.get(2).map_or("", |m| m.as_str());
                    
                    // Create timestamp key
                    let timestamp = format!("{}_{}",  date, time);
                    
                    // Add to appropriate category
                    categorized.entry(timestamp)
                        .or_insert_with(Vec::new)
                        .push(image_path.clone());
                } else {
                    // For files that don't match the pattern, put in "other" category
                    categorized.entry("other".to_string())
                        .or_insert_with(Vec::new)
                        .push(image_path.clone());
                }
            }
        }
    }
    
    Ok(categorized)
}

/// Find the highest resolution image in a group
fn find_highest_resolution(
    images: &[PathBuf]
) -> Result<(PathBuf, u32, u32, u64), Box<dyn std::error::Error>> {
    if images.is_empty() {
        return Err("No images provided to find highest resolution".into());
    }
    
    let mut best_image = images[0].clone();
    let mut best_resolution = 0;
    let mut best_width = 0;
    let mut best_height = 0;
    let mut best_size = 0;
    
    for image_path in images {
        // Get file size
        let file_size = fs::metadata(image_path)?.len();
        
        // Get image dimensions
        match image::open(image_path) {
            Ok(img) => {
                let width = img.width();
                let height = img.height();
                let resolution = width * height;
                
                // Use resolution as primary criterion, file size as secondary
                if resolution > best_resolution || 
                   (resolution == best_resolution && file_size > best_size) {
                    best_image = image_path.clone();
                    best_resolution = resolution;
                    best_width = width;
                    best_height = height;
                    best_size = file_size;
                }
            },
            Err(e) => {
                // If we can't open the image, use file size as a fallback
                if file_size > best_size {
                    best_image = image_path.clone();
                    best_resolution = 0;
                    best_width = 0;
                    best_height = 0;
                    best_size = file_size;
                    
                    println!("Warning: Could not get dimensions for {}: {}", 
                             image_path.display(), e);
                }
            }
        }
    }
    
    Ok((best_image, best_width, best_height, best_size))
}

/// Generate a text report
fn generate_report(
    entries: &[ReportEntry],
    output_path: &Path,
    processing_time: std::time::Duration
) -> Result<(), io::Error> {
    let mut file = File::create(output_path)?;
    
    writeln!(file, "Natural High Resolution Selection Report")?;
    writeln!(file, "========================================")?;
    writeln!(file, "")?;
    writeln!(file, "Total groups: {}", entries.len())?;
    writeln!(file, "Processing time: {:?}", processing_time)?;
    writeln!(file, "")?;
    writeln!(file, "Group Details:")?;
    writeln!(file, "-------------")?;
    
    for entry in entries {
        writeln!(file, "Timestamp: {}", entry.timestamp)?;
        writeln!(file, "  Images in group: {}", entry.image_count)?;
        writeln!(file, "  Selected image: {}", entry.selected_image)?;
        writeln!(file, "  Resolution: {}x{} ({:.2} MP)", 
                 entry.width, entry.height, 
                 (entry.width as f64 * entry.height as f64) / 1_000_000.0)?;
        writeln!(file, "  File size: {:.2} MB", 
                 entry.file_size as f64 / (1024.0 * 1024.0))?;
        writeln!(file, "")?;
    }
    
    Ok(())
}

/// Generate a CSV report
fn generate_csv_report(
    entries: &[ReportEntry],
    output_path: &Path
) -> Result<(), io::Error> {
    let mut file = File::create(output_path)?;
    
    // Write header
    writeln!(file, "Timestamp,Image Count,Selected Image,Width,Height,Resolution (MP),File Size (MB)")?;
    
    // Write data rows
    for entry in entries {
        let resolution_mp = (entry.width as f64 * entry.height as f64) / 1_000_000.0;
        let size_mb = entry.file_size as f64 / (1024.0 * 1024.0);
        
        writeln!(file, "{},{},\"{}\",{},{},{:.2},{:.2}",
                 entry.timestamp,
                 entry.image_count,
                 entry.selected_image,
                 entry.width,
                 entry.height,
                 resolution_mp,
                 size_mb)?;
    }
    
    Ok(())
}

/// Check if a file is a supported image type
fn is_supported_image_file(path: &Path, config: &Config) -> bool {
    if let Some(extension) = path.extension() {
        if let Some(ext_str) = extension.to_str() {
            return config.supported_formats.iter().any(|format| 
                format.eq_ignore_ascii_case(ext_str)
            );
        }
    }
    false
} 