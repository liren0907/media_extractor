use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use dashmap::DashMap;
use image::GenericImageView;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use regex::Regex;

use crate::config::Config;
use crate::image_processing::calculate_enhanced_similarity;

/// An image cache to avoid loading the same image multiple times
struct ImageCache {
    cache: DashMap<PathBuf, Arc<image::DynamicImage>>,
}

impl ImageCache {
    fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    fn get_or_load(&self, path: &Path) -> Result<Arc<image::DynamicImage>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(img) = self.cache.get(path) {
            Ok(img.clone())
        } else {
            let img = Arc::new(image::open(path)?);
            self.cache.insert(path.to_path_buf(), img.clone());
            Ok(img)
        }
    }
    
    /// Get or load an image and check if it's horizontal (if required)
    fn get_or_load_filtered(&self, path: &Path, filter_horizontal: bool) -> Result<Option<Arc<image::DynamicImage>>, Box<dyn std::error::Error + Send + Sync>> {
        let img = self.get_or_load(path)?;
        
        // Check if horizontal filtering is enabled
        if filter_horizontal {
            let (width, height) = img.dimensions();
            if width <= height {
                return Ok(None); // Skip non-horizontal images
            }
        }
        
        Ok(Some(img))
    }
}

/// Run natural high resolution split mode
pub fn run_natural_high_resolution_split(
    config: &Config,
    source_dir: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    println!("Running Natural High Resolution Split mode");
    println!("Source directory: {}", source_dir);
    
    // Check multi-threading settings
    let use_multi_threading = config.mode_options.natural_high_resolution_split.use_multi_threading;
    let thread_count = config.mode_options.natural_high_resolution_split.thread_count;
    let filter_horizontal = config.filter_horizontal_only;
    
    if filter_horizontal {
        println!("Horizontal-only filtering is enabled");
    }
    
    if use_multi_threading {
        println!("Using multi-threading with {} threads", thread_count);
        // Set the thread pool size
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build_global()
            .unwrap_or_else(|e| println!("Warning: Failed to set thread pool size: {}", e));
    } else {
        println!("Using single-threaded mode");
    }
    
    // Validate source directory
    let source_path = Path::new(source_dir);
    if !source_path.exists() || !source_path.is_dir() {
        return Err(format!("Source directory does not exist or is not a directory: {}", source_dir).into());
    }
    
    // Create output directory
    let dir_name = source_path.file_name()
        .unwrap_or_default()
        .to_string_lossy();
    
    let output_base_dir = config.get_output_dir().join("natural_high_resolution_split").join(&*dir_name);
    fs::create_dir_all(&output_base_dir)?;
    
    // Create image cache for pre-filtering
    let image_cache = Arc::new(ImageCache::new());
    
    // Get all image files
    let image_paths: Vec<PathBuf> = fs::read_dir(source_path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            if !path.is_file() || !is_supported_image_file(&path, config) {
                return false;
            }
            
            // Early filtering for horizontal-only if enabled
            if filter_horizontal {
                // We'll use the cache to avoid loading the same image twice
                match image_cache.get_or_load_filtered(&path, true) {
                    Ok(Some(_)) => true,  // Image is horizontal
                    Ok(None) => false,    // Image is not horizontal
                    Err(_) => false       // Error loading image
                }
            } else {
                true
            }
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
    
    // Create shared structures
    let report_entries = Arc::new(Mutex::new(Vec::new()));
    let total_selected = Arc::new(Mutex::new(0));
    
    // Process each timestamp group based on threading mode
    if use_multi_threading {
        process_groups_parallel(
            &categorized_images, 
            config, 
            &output_base_dir, 
            report_entries.clone(), 
            total_selected.clone(),
            image_cache
        )?;
    } else {
        process_groups_sequential(
            &categorized_images, 
            config, 
            &output_base_dir, 
            &mut report_entries.lock(), 
            &mut total_selected.lock(),
            filter_horizontal
        )?;
    }
    
    // Save report
    let report_path = output_base_dir.join("selection_report.txt");
    let mut report_file = File::create(&report_path)?;
    writeln!(report_file, "Natural High Resolution Split Selection Report")?;
    writeln!(report_file, "----------------------------------------")?;
    writeln!(report_file, "Total images processed: {}", image_paths.len())?;
    writeln!(report_file, "Total images selected: {}", *total_selected.lock())?;
    writeln!(report_file, "Multi-threading: {}", if use_multi_threading { "Enabled" } else { "Disabled" })?;
    if use_multi_threading {
        writeln!(report_file, "Thread count: {}", thread_count)?;
    }
    writeln!(report_file, "Horizontal-only filtering: {}", if filter_horizontal { "Enabled" } else { "Disabled" })?;
    writeln!(report_file, "\nDetailed Selection Report:")?;
    
    for entry in report_entries.lock().iter() {
        writeln!(report_file, "{}", entry)?;
    }
    
    println!("\nProcessing completed!");
    println!("Total images processed: {}", image_paths.len());
    println!("Total images selected: {}", *total_selected.lock());
    println!("Results saved to: {}", output_base_dir.display());
    println!("Report saved to: {}", report_path.display());
    println!("Processing time: {:?}", start_time.elapsed());
    
    Ok(())
}

/// Process timestamp groups in parallel
fn process_groups_parallel(
    categorized_images: &HashMap<String, Vec<PathBuf>>,
    config: &Config,
    output_base_dir: &Path,
    report_entries: Arc<Mutex<Vec<String>>>,
    total_selected: Arc<Mutex<usize>>,
    image_cache: Arc<ImageCache>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Process timestamp groups in parallel using for_each instead of try_for_each
    let errors = Mutex::new(Vec::new());
    let filter_horizontal = config.filter_horizontal_only;
    
    categorized_images.par_iter().for_each(|(timestamp, images)| {
        println!("\nProcessing timestamp group: {}", timestamp);
        println!("Group size: {} images", images.len());
        
        // Skip single-image groups
        if images.len() <= 1 {
            if let Some(image_path) = images.first() {
                if let Err(e) = copy_to_output(image_path, output_base_dir).map(|_| {
                    let mut total = total_selected.lock();
                    *total += 1;
                }) {
                    errors.lock().push(format!("Error copying {}: {}", image_path.display(), e));
                }
            }
            return;
        }
        
        // Find clusters within the timestamp group using parallel algorithm
        match find_similar_clusters_parallel(images, config, image_cache.clone()) {
            Ok(clusters) => {
                println!("Found {} clusters in this group", clusters.len());
                
                // Process each cluster
                for (cluster_idx, cluster) in clusters.iter().enumerate() {
                    println!("Processing cluster {} with {} images", cluster_idx + 1, cluster.len());
                    
                    // Find highest resolution image in the cluster
                    match find_highest_resolution(cluster, filter_horizontal, &image_cache) {
                        Ok(Some(highest_res)) => {
                            match copy_to_output(&highest_res, output_base_dir) {
                                Ok(_) => {
                                    // Add to report
                                    let entry = format!(
                                        "Selected {} from cluster {} in timestamp group {}",
                                        highest_res.file_name().unwrap_or_default().to_string_lossy(),
                                        cluster_idx + 1,
                                        timestamp
                                    );
                                    
                                    report_entries.lock().push(entry);
                                    
                                    // Increment counter
                                    let mut total = total_selected.lock();
                                    *total += 1;
                                },
                                Err(e) => {
                                    errors.lock().push(format!("Error copying {}: {}", highest_res.display(), e));
                                }
                            }
                        },
                        Ok(None) => {
                            errors.lock().push(format!("No highest resolution image found in cluster {} for timestamp {}", cluster_idx + 1, timestamp));
                        },
                        Err(e) => {
                            errors.lock().push(format!("Error finding highest resolution image: {}", e));
                        }
                    }
                }
            },
            Err(e) => {
                errors.lock().push(format!("Error finding clusters for timestamp {}: {}", timestamp, e));
            }
        }
    });
    
    // Check if there were any errors
    let error_list = errors.lock();
    if !error_list.is_empty() {
        return Err(format!("Errors during parallel processing:\n{}", error_list.join("\n")).into());
    }
    
    Ok(())
}

/// Process timestamp groups sequentially
fn process_groups_sequential(
    categorized_images: &HashMap<String, Vec<PathBuf>>,
    config: &Config,
    output_base_dir: &Path,
    report_entries: &mut Vec<String>,
    total_selected: &mut usize,
    filter_horizontal: bool
) -> Result<(), Box<dyn std::error::Error>> {
    for (timestamp, images) in categorized_images {
        println!("\nProcessing timestamp group: {}", timestamp);
        println!("Group size: {} images", images.len());
        
        // Skip single-image groups
        if images.len() <= 1 {
            if let Some(image_path) = images.first() {
                copy_to_output(image_path, output_base_dir)?;
                *total_selected += 1;
            }
            continue;
        }
        
        // Find clusters within the timestamp group
        let clusters = find_similar_clusters_sequential(images, config, filter_horizontal)?;
        println!("Found {} clusters in this group", clusters.len());
        
        // Process each cluster
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            println!("Processing cluster {} with {} images", cluster_idx + 1, cluster.len());
            
            // Find highest resolution image in the cluster
            if let Some(highest_res) = find_highest_resolution_sequential(cluster, filter_horizontal)? {
                let _dest_path = copy_to_output(&highest_res, output_base_dir)?;
                report_entries.push(format!(
                    "Selected {} from cluster {} in timestamp group {}",
                    highest_res.file_name().unwrap_or_default().to_string_lossy(),
                    cluster_idx + 1,
                    timestamp
                ));
                *total_selected += 1;
            }
        }
    }
    
    Ok(())
}

/// Find clusters of similar images within a group (parallel version)
fn find_similar_clusters_parallel(
    images: &[PathBuf],
    config: &Config,
    image_cache: Arc<ImageCache>,
) -> Result<Vec<Vec<PathBuf>>, Box<dyn std::error::Error + Send + Sync>> {
    let n = images.len();
    let processed = Arc::new(RwLock::new(vec![false; n]));
    let clusters = Arc::new(Mutex::new(Vec::new()));
    let similarity_threshold = config.mode_options.natural_high_resolution_split.split_similarity_threshold;
    let filter_horizontal = config.filter_horizontal_only;
    let error_flag = Arc::new(Mutex::new(None));
    
    (0..n).into_par_iter().for_each(|i| {
        // Skip if already in an error state
        if error_flag.lock().is_some() {
            return;
        }
        
        // Skip if already processed
        {
            let processed_guard = processed.read();
            if processed_guard[i] {
                return;
            }
        }
        
        // Start a new cluster
        let mut current_cluster = vec![images[i].clone()];
        
        // Mark as processed
        {
            let mut processed_guard = processed.write();
            processed_guard[i] = true;
        }
        
        // Load the reference image with horizontal filtering if needed
        let img_i = match image_cache.get_or_load_filtered(&images[i], filter_horizontal) {
            Ok(Some(img)) => img,
            Ok(None) => {
                // Skip this image if it's filtered out (not horizontal but filtering is enabled)
                return;
            },
            Err(e) => {
                let mut err_guard = error_flag.lock();
                if err_guard.is_none() {
                    *err_guard = Some(format!("Failed to load image {}: {}", images[i].display(), e));
                }
                return;
            }
        };
        
        // Compare with remaining images
        for j in (i + 1)..n {
            // Skip if already in an error state
            if error_flag.lock().is_some() {
                return;
            }
            
            // Skip if already processed
            {
                let processed_guard = processed.read();
                if processed_guard[j] {
                    continue;
                }
            }
            
            // Load comparison image with horizontal filtering if needed
            let img_j = match image_cache.get_or_load_filtered(&images[j], filter_horizontal) {
                Ok(Some(img)) => img,
                Ok(None) => {
                    // Skip this image if it's filtered out (not horizontal but filtering is enabled)
                    continue;
                },
                Err(e) => {
                    let mut err_guard = error_flag.lock();
                    if err_guard.is_none() {
                        *err_guard = Some(format!("Failed to load image {}: {}", images[j].display(), e));
                    }
                    continue;
                }
            };
            
            // Calculate similarity
            let similarity = calculate_enhanced_similarity(&img_i, &img_j, config);
            
            // Add to cluster if similar
            if similarity >= similarity_threshold {
                current_cluster.push(images[j].clone());
                
                // Mark as processed
                let mut processed_guard = processed.write();
                processed_guard[j] = true;
            }
        }
        
        // Add the cluster
        if !current_cluster.is_empty() {
            clusters.lock().push(current_cluster);
        }
    });
    
    // Check for errors
    if let Some(err) = error_flag.lock().as_ref() {
        return Err(err.clone().into());
    }
    
    // Return the clusters
    Ok(Arc::try_unwrap(clusters)
        .map(|mutex| mutex.into_inner())
        .unwrap_or_else(|arc_mutex| arc_mutex.lock().clone()))
}

/// Find clusters of similar images within a group (sequential version)
fn find_similar_clusters_sequential(
    images: &[PathBuf],
    config: &Config,
    filter_horizontal: bool
) -> Result<Vec<Vec<PathBuf>>, Box<dyn std::error::Error>> {
    let mut clusters: Vec<Vec<PathBuf>> = Vec::new();
    let mut processed: Vec<bool> = vec![false; images.len()];
    let similarity_threshold = config.mode_options.natural_high_resolution_split.split_similarity_threshold;
    
    for i in 0..images.len() {
        if processed[i] {
            continue;
        }
        
        let mut current_cluster = vec![images[i].clone()];
        processed[i] = true;
        
        // Load the reference image
        let img_i = match image::open(&images[i]) {
            Ok(img) => {
                // Skip if horizontal filtering is enabled and image is not horizontal
                if filter_horizontal {
                    let (width, height) = img.dimensions();
                    if width <= height {
                        continue; // Skip non-horizontal images
                    }
                }
                img
            },
            Err(_) => continue, // Skip failed images
        };
        
        for j in (i + 1)..images.len() {
            if processed[j] {
                continue;
            }
            
            // Load comparison image
            let img_j = match image::open(&images[j]) {
                Ok(img) => {
                    // Skip if horizontal filtering is enabled and image is not horizontal
                    if filter_horizontal {
                        let (width, height) = img.dimensions();
                        if width <= height {
                            continue; // Skip non-horizontal images
                        }
                    }
                    img
                },
                Err(_) => continue, // Skip failed images
            };
            
            let similarity = calculate_enhanced_similarity(&img_i, &img_j, config);
            
            if similarity >= similarity_threshold {
                current_cluster.push(images[j].clone());
                processed[j] = true;
            }
        }
        
        if !current_cluster.is_empty() {
            clusters.push(current_cluster);
        }
    }
    
    Ok(clusters)
}

/// Find the image with the highest resolution in a cluster (parallel version)
fn find_highest_resolution(
    images: &[PathBuf], 
    filter_horizontal: bool,
    image_cache: &Arc<ImageCache>
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    if images.is_empty() {
        return Ok(None);
    }
    
    // Use a thread-safe structure to collect image dimensions
    let image_dimensions = DashMap::new();
    let errors = Mutex::new(Vec::new());
    
    // First, collect all image dimensions in parallel
    images.par_iter().for_each(|image_path| {
        let img_result = match filter_horizontal {
            true => image_cache.get_or_load_filtered(image_path, true),
            false => image_cache.get_or_load(image_path).map(Some),
        };
        
        match img_result {
            Ok(Some(img)) => {
                let dimensions = img.dimensions();
                let total_pixels = dimensions.0 * dimensions.1;
                image_dimensions.insert(image_path.clone(), total_pixels);
            },
            Ok(None) => {}, // Image was filtered out (not horizontal)
            Err(e) => {
                errors.lock().push(format!("Error loading image {}: {}", image_path.display(), e));
            }
        }
    });
    
    // Check if there were any errors
    let error_list = errors.lock();
    if !error_list.is_empty() {
        eprintln!("Warnings during image resolution checking: {}", error_list.join(", "));
    }
    
    // Find the image with highest resolution
    let mut highest_res = None;
    let mut max_pixels = 0;
    
    for item in image_dimensions.iter() {
        let path = item.key();
        let pixels = *item.value();
        
        if pixels > max_pixels {
            max_pixels = pixels;
            highest_res = Some(path.clone());
        }
    }
    
    Ok(highest_res)
}

/// Find the image with the highest resolution in a cluster (sequential version)
fn find_highest_resolution_sequential(
    images: &[PathBuf],
    filter_horizontal: bool
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let mut highest_res = None;
    let mut max_pixels = 0;
    
    for image_path in images {
        if let Ok(img) = image::open(image_path) {
            // Apply horizontal filtering if enabled
            if filter_horizontal {
                let (width, height) = img.dimensions();
                if width <= height {
                    continue; // Skip non-horizontal images
                }
            }
            
            let dimensions = img.dimensions();
            let total_pixels = dimensions.0 * dimensions.1;
            
            if total_pixels > max_pixels {
                max_pixels = total_pixels;
                highest_res = Some(image_path.clone());
            }
        }
    }
    
    Ok(highest_res)
}

/// Copy an image to the output directory
fn copy_to_output(source: &Path, output_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file_name = source.file_name().ok_or("Invalid file name")?;
    let dest_path = output_dir.join(file_name);
    fs::copy(source, &dest_path)?;
    Ok(dest_path)
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

/// Categorize images by timestamp from filename
fn categorize_by_timestamp(image_paths: &[PathBuf]) -> Result<HashMap<String, Vec<PathBuf>>, Box<dyn std::error::Error>> {
    let mut categorized = HashMap::new();
    let timestamp_pattern = Regex::new(r"\d{8}_\d{6}")?;
    
    for path in image_paths {
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            if let Some(timestamp) = timestamp_pattern.find(&file_name_str) {
                categorized
                    .entry(timestamp.as_str().to_string())
                    .or_insert_with(Vec::new)
                    .push(path.clone());
            }
        }
    }
    
    Ok(categorized)
} 