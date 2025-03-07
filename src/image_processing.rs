use image::{DynamicImage, imageops::FilterType, ImageError, GenericImageView};
use std::path::Path;
use crate::config::{Config, MatchingMethod};

#[cfg(feature = "opencv")]
use {
    opencv::{
        prelude::*,
        core,
        features2d::{self, Feature2D, SIFT, SURF},
        imgproc,
        imgcodecs,
    },
    std::sync::Once,
};

// Global OpenCV initialization
#[cfg(feature = "opencv")]
static OPENCV_INIT: Once = Once::new();

#[cfg(feature = "opencv")]
fn init_opencv() {
    OPENCV_INIT.call_once(|| {
        opencv::core::set_num_threads(0).ok(); // Use all available cores
    });
}

/// Compare images using SIFT features if OpenCV is enabled
#[cfg(feature = "opencv")]
pub fn compare_with_sift(img1: &DynamicImage, img2: &DynamicImage, config: &Config) -> Result<f64, ImageError> {
    init_opencv();
    
    // Convert DynamicImage to OpenCV Mat
    let img1_mat = dynamic_image_to_mat(img1)?;
    let img2_mat = dynamic_image_to_mat(img2)?;
    
    // Convert to grayscale for feature detection
    let mut gray1 = Mat::default();
    let mut gray2 = Mat::default();
    imgproc::cvt_color(&img1_mat, &mut gray1, imgproc::COLOR_BGR2GRAY, 0)?;
    imgproc::cvt_color(&img2_mat, &mut gray2, imgproc::COLOR_BGR2GRAY, 0)?;
    
    // Create SIFT detector
    let feature_count = config.sift_feature_count;
    let mut sift = SIFT::create(feature_count, 3, 0.04, 10.0, 1.6)?;
    
    // Detect keypoints and compute descriptors
    let mut keypoints1 = Vector::new();
    let mut descriptors1 = Mat::default();
    sift.detect_and_compute(&gray1, &core::no_array(), &mut keypoints1, &mut descriptors1, false)?;
    
    let mut keypoints2 = Vector::new();
    let mut descriptors2 = Mat::default();
    sift.detect_and_compute(&gray2, &core::no_array(), &mut keypoints2, &mut descriptors2, false)?;
    
    // If no keypoints found in either image, return 0 similarity
    if keypoints1.is_empty() || keypoints2.is_empty() || descriptors1.empty() || descriptors2.empty() {
        return Ok(0.0);
    }
    
    // Match descriptors using BFMatcher
    let mut matcher = features2d::BFMatcher::create(core::NORM_L2, false)?;
    let matches = matcher.match_(&descriptors1, &descriptors2, &core::no_array())?;
    
    // Filter good matches
    let mut good_matches = Vector::new();
    let mut min_dist = std::f32::MAX;
    
    // Find min distance
    for m in matches.iter() {
        if m.distance < min_dist {
            min_dist = m.distance;
        }
    }
    
    // Keep only good matches (distance < 2*min_dist)
    let max_dist = 3.0 * min_dist;
    for m in matches.iter() {
        if m.distance <= max_dist {
            good_matches.push(m);
        }
    }
    
    // Calculate similarity score based on % of keypoints matched
    let match_count = good_matches.len();
    let total_keypoints = keypoints1.len().min(keypoints2.len());
    
    let similarity = if total_keypoints > 0 {
        let ratio = match_count as f64 / total_keypoints as f64;
        // Apply tolerance similar to how we do for hash-based methods
        let min_threshold = config.sift_match_threshold * 0.7;
        if ratio >= min_threshold {
            let boost_zone = config.sift_match_threshold - min_threshold;
            if ratio < config.sift_match_threshold {
                // Apply gentle boost similar to bit error tolerance for near-matches
                min_threshold + (ratio - min_threshold) / boost_zone
            } else {
                ratio
            }
        } else {
            ratio
        }
    } else {
        0.0
    };
    
    Ok(similarity)
}

/// Compare images using SIFT features - fallback when OpenCV is not available
#[cfg(not(feature = "opencv"))]
pub fn compare_with_sift(_img1: &DynamicImage, _img2: &DynamicImage, _config: &Config) -> Result<f64, ImageError> {
    // Create a proper UnsupportedError struct
    Err(ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
        image::error::ImageFormatHint::Unknown,
        image::error::UnsupportedErrorKind::GenericFeature(
            "SIFT comparison requires OpenCV support, rebuild with 'opencv' feature enabled".to_string()
        )
    )))
}

/// Helper function to convert DynamicImage to OpenCV Mat
#[cfg(feature = "opencv")]
fn dynamic_image_to_mat(img: &DynamicImage) -> Result<Mat, ImageError> {
    let rgb_img = img.to_rgb8();
    let width = rgb_img.width() as i32;
    let height = rgb_img.height() as i32;
    let bytes = rgb_img.as_raw();
    
    // Create a Mat from the raw bytes (RGB format)
    let mut mat = unsafe {
        Mat::new_rows_cols_with_data(
            height, width,
            core::CV_8UC3,
            bytes.as_ptr() as *mut std::os::raw::c_void,
            core::Mat_AUTO_STEP
        )?
    };
    
    // Convert RGB to BGR (OpenCV default)
    let mut bgr_mat = Mat::default();
    imgproc::cvt_color(&mat, &mut bgr_mat, imgproc::COLOR_RGB2BGR, 0)?;
    
    Ok(bgr_mat)
}

/// Calculates a perceptual hash for an image
pub fn calculate_hash(img: &DynamicImage, config: &Config) -> Vec<bool> {
    calculate_hash_with_params(img, config.hash_size, config.blur_radius)
}

/// Calculates a perceptual hash for an image with specific parameters
pub fn calculate_hash_with_params(img: &DynamicImage, hash_size: u32, blur_radius: f32) -> Vec<bool> {
    // Resize image to small square (8x8 is standard for perceptual hash)
    let small = img.blur(blur_radius).resize_exact(
        hash_size, 
        hash_size, 
        FilterType::Triangle
    );
    
    // Convert to grayscale
    let gray = small.to_luma8();
    
    // Calculate average pixel value
    let mut total: u32 = 0;
    for pixel in gray.pixels() {
        total += pixel[0] as u32;
    }
    let avg = total / (hash_size * hash_size);
    
    // Generate hash (each bit is 1 if pixel >= avg, 0 if < avg)
    let mut hash = Vec::with_capacity((hash_size * hash_size) as usize);
    for pixel in gray.pixels() {
        hash.push(pixel[0] as u32 >= avg);
    }
    
    hash
}

/// Calculates a color-aware perceptual hash for an image
pub fn calculate_color_hash(img: &DynamicImage, config: &Config) -> Vec<bool> {
    calculate_color_hash_with_params(img, config.hash_size, config.blur_radius)
}

/// Calculates a color-aware perceptual hash with specific parameters
pub fn calculate_color_hash_with_params(img: &DynamicImage, hash_size: u32, blur_radius: f32) -> Vec<bool> {
    // Resize image to small square
    let small = img.blur(blur_radius).resize_exact(
        hash_size, 
        hash_size, 
        FilterType::Triangle
    );
    
    // Extract RGB channels
    let rgb = small.to_rgb8();
    
    let total_pixels = (hash_size * hash_size) as usize;
    let mut hash = Vec::with_capacity(total_pixels * 3);
    
    // Calculate average for each channel
    let mut r_total = 0;
    let mut g_total = 0;
    let mut b_total = 0;
    
    for pixel in rgb.pixels() {
        r_total += pixel[0] as u32;
        g_total += pixel[1] as u32;
        b_total += pixel[2] as u32;
    }
    
    let r_avg = r_total / (hash_size * hash_size);
    let g_avg = g_total / (hash_size * hash_size);
    let b_avg = b_total / (hash_size * hash_size);
    
    // Generate hash (each channel contributes)
    for pixel in rgb.pixels() {
        hash.push(pixel[0] as u32 >= r_avg);
        hash.push(pixel[1] as u32 >= g_avg);
        hash.push(pixel[2] as u32 >= b_avg);
    }
    
    hash
}

/// Calculates a hash that is extremely resistant to noise
/// This implementation focuses only on dominant features and color regions
pub fn calculate_noise_resistant_hash(img: &DynamicImage, config: &Config) -> Vec<bool> {
    let hash_size = config.hash_size as u32;
    
    // Step 1: Apply heavy blur to remove noise completely (higher than normal blur)
    let very_blurred = img.blur(config.blur_radius * 1.5);
    
    // Step 2: Convert to lower resolution to focus on dominant features only
    let img_small = very_blurred.resize_exact(
        hash_size, 
        hash_size, 
        FilterType::Triangle  // Faster filter that still works well for downsampling
    );
    
    // Step 3: Extract color information more aggressively
    let rgb = img_small.to_rgb8();
    let pixels = rgb.as_raw();
    
    // Pre-allocate the hash vector
    let pixel_count = (hash_size * hash_size) as usize;
    let total_size = pixel_count * 3; // R,G,B channels
    let mut hash = Vec::with_capacity(total_size);
    
    // Process each channel separately (R,G,B) with more aggressive thresholding
    for channel in 0..3 {
        // Extract channel values
        let mut channel_values = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            channel_values.push(pixels[i * 3 + channel]);
        }
        
        // Calculate median value (more robust than average for noisy images)
        let mut sorted_values = channel_values.clone();
        sorted_values.sort();
        let median = if pixel_count > 0 {
            sorted_values[pixel_count / 2]
        } else {
            128
        };
        
        // Use more aggressive thresholding - compare to median instead of average
        for &value in &channel_values {
            hash.push(value >= median);
        }
    }
    
    hash
}

/// Calculates the similarity between two image hashes
/// Returns a value between 0.0 (no similarity) and 1.0 (identical)
pub fn calculate_similarity(hash1: &[bool], hash2: &[bool]) -> f64 {
    if hash1.len() != hash2.len() || hash1.is_empty() {
        return 0.0;
    }
    
    // Count the number of matching bits
    let matching = hash1.iter()
        .zip(hash2.iter())
        .filter(|&(a, b)| a == b)
        .count();
    
    matching as f64 / hash1.len() as f64
}

/// Calculates the similarity between two image hashes with error tolerance for noisy images
/// The tolerance allows for a percentage of bits to be different due to noise
pub fn calculate_similarity_with_tolerance(hash1: &[bool], hash2: &[bool], tolerance: f64) -> f64 {
    if hash1.len() != hash2.len() || hash1.is_empty() {
        return 0.0;
    }
    
    // Skip bits can help with very noisy images by only comparing every Nth bit
    // This is especially useful with very noisy images where single bits can be corrupted
    let skip_bits = if tolerance > 0.2 { 2 } else { 1 };
    
    // Count matching bits, but only consider every Nth bit if skip_bits > 1
    let mut matching = 0;
    let mut total = 0;
    
    for (i, (a, b)) in hash1.iter().zip(hash2.iter()).enumerate() {
        if skip_bits > 1 && i % skip_bits != 0 {
            continue;  // Skip some bits for high noise tolerance
        }
        if a == b {
            matching += 1;
        }
        total += 1;
    }
    
    let similarity = if total > 0 {
        matching as f64 / total as f64
    } else {
        0.0
    };
    
    // Apply tolerance - boost similarity if it's close to the threshold
    // This helps with noisy images by giving a boost to near-matches
    if similarity > (1.0 - tolerance * 1.5) {
        let boost = (1.0 - similarity) * tolerance * 1.5;
        return (similarity + boost).min(1.0);  // Ensure we don't exceed 1.0
    }
    
    similarity
}

/// Enhanced image hash comparison that takes rotation and flipping into account
/// This is more computationally expensive but can detect similar images in different orientations
pub fn calculate_robust_similarity(hash1: &[bool], hash2: &[bool], config: &Config) -> f64 {
    let hash_size = config.hash_size as usize;
    let total_size = hash_size * hash_size;
    
    if hash1.len() != hash2.len() || hash1.len() != total_size {
        return 0.0;
    }

    // Original comparison is always done
    let original_similarity = calculate_similarity_with_tolerance(hash1, hash2, config.bit_error_tolerance);
    
    // Return early if we don't need to consider rotations or flips
    if !config.consider_rotations && !config.consider_flips {
        return original_similarity;
    }
    
    // Early return if the images are already very similar (optimization)
    if original_similarity > config.similarity_threshold {
        return original_similarity;
    }
    
    // We'll return the best similarity after considering various transformations
    let mut best_similarity = original_similarity;

    // Preallocation and reuse of memory improves performance
    // Convert 1D hash to 2D for easier rotation
    let mut grid1 = vec![vec![false; hash_size]; hash_size];
    let mut grid2 = vec![vec![false; hash_size]; hash_size];
    
    // Populate grids only once
    for i in 0..hash_size {
        for j in 0..hash_size {
            grid1[i][j] = hash1[i * hash_size + j];
            grid2[i][j] = hash2[i * hash_size + j];
        }
    }
    
    // Preallocate space for transformation outputs to avoid repeated memory allocations
    let mut transformed_hash = vec![false; total_size];
    
    if config.consider_rotations {
        // Test 90° rotation
        let rotated90 = rotate_grid(grid2.clone(), hash_size);
        flatten_grid(&rotated90, &mut transformed_hash, hash_size);
        let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
        best_similarity = best_similarity.max(similarity);
        
        // If we found a good match, early return to save computation
        if best_similarity > config.similarity_threshold {
            return best_similarity;
        }
        
        // Test 180° rotation
        let rotated180 = rotate_grid(rotated90, hash_size);
        flatten_grid(&rotated180, &mut transformed_hash, hash_size);
        let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
        best_similarity = best_similarity.max(similarity);
        
        if best_similarity > config.similarity_threshold {
            return best_similarity;
        }
        
        // Test 270° rotation
        let rotated270 = rotate_grid(rotated180, hash_size);
        flatten_grid(&rotated270, &mut transformed_hash, hash_size);
        let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
        best_similarity = best_similarity.max(similarity);
        
        if best_similarity > config.similarity_threshold {
            return best_similarity;
        }
    }
    
    if config.consider_flips {
        // Test horizontal flip
        let h_flipped = flip_horizontal(&grid2, hash_size);
        flatten_grid(&h_flipped, &mut transformed_hash, hash_size);
        let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
        best_similarity = best_similarity.max(similarity);
        
        if best_similarity > config.similarity_threshold {
            return best_similarity;
        }
        
        // If we also consider rotations, test rotations of the flipped hash
        if config.consider_rotations {
            // Test flipped + 90°
            let flipped_rotated90 = rotate_grid(h_flipped.clone(), hash_size);
            flatten_grid(&flipped_rotated90, &mut transformed_hash, hash_size);
            let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
            best_similarity = best_similarity.max(similarity);
            
            if best_similarity > config.similarity_threshold {
                return best_similarity;
            }
            
            // Test flipped + 180°
            let flipped_rotated180 = rotate_grid(flipped_rotated90, hash_size);
            flatten_grid(&flipped_rotated180, &mut transformed_hash, hash_size);
            let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
            best_similarity = best_similarity.max(similarity);
            
            if best_similarity > config.similarity_threshold {
                return best_similarity;
            }
            
            // Test flipped + 270°
            let flipped_rotated270 = rotate_grid(flipped_rotated180, hash_size);
            flatten_grid(&flipped_rotated270, &mut transformed_hash, hash_size);
            let similarity = calculate_similarity_with_tolerance(hash1, &transformed_hash, config.bit_error_tolerance);
            best_similarity = best_similarity.max(similarity);
        }
    }
    
    best_similarity
}

// Helper function to rotate a grid 90 degrees clockwise
fn rotate_grid(grid: Vec<Vec<bool>>, size: usize) -> Vec<Vec<bool>> {
    let mut rotated = vec![vec![false; size]; size];
    for i in 0..size {
        for j in 0..size {
            rotated[j][size - 1 - i] = grid[i][j];
        }
    }
    rotated
}

// Helper function to flip a grid horizontally
fn flip_horizontal(grid: &[Vec<bool>], size: usize) -> Vec<Vec<bool>> {
    let mut flipped = vec![vec![false; size]; size];
    for i in 0..size {
        for j in 0..size {
            flipped[i][size - 1 - j] = grid[i][j];
        }
    }
    flipped
}

// Helper function to flatten a grid to 1D
fn flatten_grid(grid: &[Vec<bool>], output: &mut [bool], size: usize) {
    for i in 0..size {
        for j in 0..size {
            output[i * size + j] = grid[i][j];
        }
    }
}

/// Compute an image hash from a file path
pub fn compute_image_hash(path: &Path, _hash_size: u32, _filter_type: FilterType) -> Result<u64, ImageError> {
    let img = image::open(path)?;
    let hash_size = 8; // Default value for compatibility
    
    // Convert to grayscale for speed
    let img = img.grayscale();
    
    // Resize the image
    let small = img.resize_exact(hash_size, hash_size, FilterType::Lanczos3);
    
    // Convert to luma
    let luma = small.to_luma8();
    
    // Calculate average value
    let mut avg: u32 = 0;
    for p in luma.pixels() {
        avg += p[0] as u32;
    }
    avg /= (hash_size * hash_size) as u32;
    
    // Compute the hash
    let mut hash: u64 = 0;
    for (i, p) in luma.pixels().enumerate() {
        if p[0] as u32 >= avg {
            hash |= 1 << i;
        }
    }
    
    Ok(hash)
}

/// Calculates a structural similarity index for extremely noisy images
/// This function is designed to work when traditional perceptual hashing fails
/// It compares the overall structure and color distributions rather than exact pixels
pub fn calculate_structural_similarity_with_params(
    img1: &DynamicImage, 
    img2: &DynamicImage, 
    blur_radius: f32
) -> Result<f64, ImageError> {
    // Resize images to a common size if they differ
    let (width1, height1) = img1.dimensions();
    let (width2, height2) = img2.dimensions();
    
    // Calculate aspect ratios
    let aspect1 = width1 as f32 / height1 as f32;
    let aspect2 = width2 as f32 / height2 as f32;
    
    // Aspect similarity - ranges from 0 to 1 where 1 means identical aspect ratio
    let aspect_similarity: f64 = (1.0 - (aspect1 - aspect2).abs() / (aspect1 + aspect2)).max(0.3) as f64;
    
    // If images are different sizes, resize to smallest
    let (max_width, max_height) = (width1.min(width2), height1.min(height2));
    let target_width = max_width.min(512); // Cap at 512 for performance
    let target_height = max_height.min(512);
    
    let img1_resized = img1.resize_exact(target_width, target_height, FilterType::Lanczos3);
    let img2_resized = img2.resize_exact(target_width, target_height, FilterType::Lanczos3);
    
    // Convert to grayscale
    let gray1 = img1_resized.to_luma8();
    let gray2 = img2_resized.to_luma8();
    
    // Constants for SSIM calculation
    let k1: f64 = 0.01;
    let k2: f64 = 0.03;
    let l: f64 = 255.0; // Dynamic range of pixel values
    
    // Instead of using imageproc::filter::gaussian_blur_f32 which has compatibility issues,
    // we'll use the image crate's built-in blur functionality
    let gray1_blurred = DynamicImage::ImageLuma8(gray1.clone()).blur(blur_radius as f32);
    let gray2_blurred = DynamicImage::ImageLuma8(gray2.clone()).blur(blur_radius as f32);
    
    // Convert blurred images back to luma8
    let gray1_blurred = gray1_blurred.to_luma8();
    let gray2_blurred = gray2_blurred.to_luma8();
    
    // Calculate mean, variance, and covariance
    let mut mean1: f64 = 0.0;
    let mut mean2: f64 = 0.0;
    let mut var1: f64 = 0.0;
    let mut var2: f64 = 0.0;
    let mut covar: f64 = 0.0;
    let total_pixels = (target_width * target_height) as f64;
    
    for (p1, p2) in gray1_blurred.iter().zip(gray2_blurred.iter()) {
        let pixel1 = *p1 as f64;
        let pixel2 = *p2 as f64;
        
        mean1 += pixel1;
        mean2 += pixel2;
    }
    
    mean1 /= total_pixels;
    mean2 /= total_pixels;
    
    for (p1, p2) in gray1_blurred.iter().zip(gray2_blurred.iter()) {
        let pixel1 = *p1 as f64;
        let pixel2 = *p2 as f64;
        
        let diff1 = pixel1 - mean1;
        let diff2 = pixel2 - mean2;
        
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        covar += diff1 * diff2;
    }
    
    var1 /= total_pixels;
    var2 /= total_pixels;
    covar /= total_pixels;
    
    // Calculate SSIM constants
    let c1 = (k1 * l).powi(2);
    let c2 = (k2 * l).powi(2);
    
    // Adjust covariance by aspect similarity (optional)
    covar = covar * aspect_similarity;
    
    // Calculate SSIM
    let ssim = ((2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2)) / 
               ((mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2));
    
    // Map SSIM range from [-1,1] to [0,1] for consistency with our other similarity measures
    let normalized_ssim = (ssim + 1.0) / 2.0;
    
    Ok(normalized_ssim)
}

/// Compares just the average color of two images - for extreme cases
/// This is essentially a 1x1 hash comparison
fn compare_average_colors(img1: &DynamicImage, img2: &DynamicImage, tolerance: f64) -> f64 {
    // Apply extreme blur to remove all details
    let blurred1 = img1.blur(10.0);  // Very high blur radius
    let blurred2 = img2.blur(10.0);
    
    // Resize to 1x1 to get average color
    let tiny1 = blurred1.resize_exact(1, 1, FilterType::Triangle);
    let tiny2 = blurred2.resize_exact(1, 1, FilterType::Triangle);
    
    // Get RGB values - fix the temporary value issue
    let rgb1_img = tiny1.to_rgb8();
    let rgb1 = rgb1_img.get_pixel(0, 0);
    
    let rgb2_img = tiny2.to_rgb8();
    let rgb2 = rgb2_img.get_pixel(0, 0);
    
    // Calculate color difference normalized to [0,1]
    let r_diff = (rgb1[0] as f64 - rgb2[0] as f64).abs() / 255.0;
    let g_diff = (rgb1[1] as f64 - rgb2[1] as f64).abs() / 255.0;
    let b_diff = (rgb1[2] as f64 - rgb2[2] as f64).abs() / 255.0;
    
    // Average difference across all channels
    let avg_diff = (r_diff + g_diff + b_diff) / 3.0;
    
    // Convert to similarity (1.0 - diff) with extra tolerance boost
    let mut similarity = 1.0 - avg_diff;
    
    // Apply tolerance - very aggressively
    if similarity > (0.3) {  // If there's even a slight resemblance
        let boost = (1.0 - similarity) * tolerance * 2.0;
        similarity = (similarity + boost).min(1.0);
    }
    
    similarity
}

/// Calculate similarity between two images using the specified method
pub fn calculate_image_similarity_with_method(img1: &DynamicImage, img2: &DynamicImage, config: &Config) -> f64 {
    match config.matching_method {
        MatchingMethod::PerceptualHash => {
            // Use the existing perceptual hash method
            let hash1 = calculate_hash_with_params(img1, config.hash_size, config.blur_radius);
            let hash2 = calculate_hash_with_params(img2, config.hash_size, config.blur_radius);
            calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
        },
        MatchingMethod::ColorHash => {
            // Use the color hash method
            let hash1 = calculate_color_hash_with_params(img1, config.hash_size, config.blur_radius);
            let hash2 = calculate_color_hash_with_params(img2, config.hash_size, config.blur_radius);
            calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
        },
        MatchingMethod::Sift => {
            // Use SIFT feature matching
            match compare_with_sift(img1, img2, config) {
                Ok(similarity) => similarity,
                Err(e) => {
                    // Fallback to perceptual hash on error
                    log::warn!("SIFT matching failed: {}, falling back to perceptual hash", e);
                    let hash1 = calculate_hash_with_params(img1, config.hash_size, config.blur_radius);
                    let hash2 = calculate_hash_with_params(img2, config.hash_size, config.blur_radius);
                    calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
                }
            }
        },
        MatchingMethod::Surf => {
            // For now, fall back to SIFT (we can implement SURF separately later)
            match compare_with_sift(img1, img2, config) {
                Ok(similarity) => similarity,
                Err(e) => {
                    log::warn!("SURF matching failed: {}, falling back to perceptual hash", e);
                    let hash1 = calculate_hash_with_params(img1, config.hash_size, config.blur_radius);
                    let hash2 = calculate_hash_with_params(img2, config.hash_size, config.blur_radius);
                    calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
                }
            }
        },
        MatchingMethod::Ssim => {
            // Use structural similarity
            match calculate_structural_similarity_with_params(img1, img2, config.blur_radius) {
                Ok(similarity) => similarity,
                Err(_) => {
                    // Fallback to perceptual hash on error
                    let hash1 = calculate_hash_with_params(img1, config.hash_size, config.blur_radius);
                    let hash2 = calculate_hash_with_params(img2, config.hash_size, config.blur_radius);
                    calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance)
                }
            }
        }
    }
}

/// Compare two images and return their similarity
pub fn compare_images(path1: &str, path2: &str, config: &Config) -> Result<f64, Box<dyn std::error::Error>> {
    // Load the images
    let img1 = image::open(Path::new(path1))?;
    let img2 = image::open(Path::new(path2))?;

    // Use the universal matching algorithm based on config
    let similarity = calculate_image_similarity_with_method(&img1, &img2, config);
    
    println!("Similarity between images: {:.2}%", similarity * 100.0);

    Ok(similarity)
}

/// Enhanced image matching that combines multiple techniques for better noise resistance
pub fn calculate_enhanced_similarity(img1: &DynamicImage, img2: &DynamicImage, config: &Config) -> f64 {
    // Apply stronger blur for noise reduction
    let enhanced_blur = config.blur_radius * 1.5;
    
    // 1. Try perceptual hash with increased blur
    let hash1 = calculate_hash_with_params(img1, config.hash_size, enhanced_blur);
    let hash2 = calculate_hash_with_params(img2, config.hash_size, enhanced_blur);
    let hash_similarity = calculate_similarity_with_tolerance(&hash1, &hash2, config.bit_error_tolerance);
    
    // 2. Try color-aware hash if color is important
    let mut color_similarity = 0.0;
    if config.use_color_hash {
        let color_hash1 = calculate_color_hash_with_params(img1, config.hash_size, enhanced_blur);
        let color_hash2 = calculate_color_hash_with_params(img2, config.hash_size, enhanced_blur);
        color_similarity = calculate_similarity(&color_hash1, &color_hash2);
    }
    
    // 3. Try structural similarity
    let mut structural_similarity = 0.0;
    let downsized_img1 = img1.resize(64, 64, image::imageops::FilterType::Lanczos3);
    let downsized_img2 = img2.resize(64, 64, image::imageops::FilterType::Lanczos3);
    structural_similarity = match calculate_structural_similarity_with_params(
        &downsized_img1, &downsized_img2, enhanced_blur) {
        Ok(similarity) => similarity,
        Err(_) => 0.0 // Fall back to 0 on error
    };
    
    // Calculate weighted similarity score
    let mut final_score = 0.0;
    let mut weight_sum = 0.0;
    
    // Add hash similarity with highest weight
    final_score += hash_similarity * 0.5;
    weight_sum += 0.5;
    
    // Add color similarity if used
    if config.use_color_hash {
        final_score += color_similarity * 0.2;
        weight_sum += 0.2;
    }
    
    // Add structural similarity
    final_score += structural_similarity * 0.3;
    weight_sum += 0.3;
    
    // Normalize
    if weight_sum > 0.0 {
        final_score /= weight_sum;
    }
    
    // Apply additional boost for near-threshold matches
    if final_score >= config.similarity_threshold * 0.8 && final_score < config.similarity_threshold {
        // Boost scores that are close to threshold
        final_score = final_score * 1.1;
        // Cap at 1.0
        if final_score > 1.0 {
            final_score = 1.0;
        }
    }
    
    // Log detailed component scores for debugging
    println!("  Components: pHash: {:.2}%, ColorHash: {:.2}%, SSIM: {:.2}%", 
        hash_similarity * 100.0,
        color_similarity * 100.0,
        structural_similarity * 100.0);
    
    final_score
}
