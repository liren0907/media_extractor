use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use image::GenericImageView;

use crate::config::Config;
use crate::image_processing::calculate_enhanced_similarity;

/// Result structure for confusion matrix
pub struct ConfusionMatrixResult {
    pub image_names: Vec<String>,
    pub similarity_matrix: Vec<Vec<f64>>,
    pub processing_time: std::time::Duration,
}

/// Generate a confusion matrix for all images in a directory
pub fn generate_confusion_matrix(
    source_dir: &Path,
    config: &Config,
) -> Result<ConfusionMatrixResult, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    println!("Generating confusion matrix for images in: {}", source_dir.display());
    println!("Using matching method: {:?}", config.matching_method);
    
    // Get all image files in the directory
    let image_paths: Vec<PathBuf> = fs::read_dir(source_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && is_supported_image_file(&path, config)
        })
        .map(|entry| entry.path())
        .collect();
    
    if image_paths.is_empty() {
        return Err(format!("No supported images found in {}", source_dir.display()).into());
    }
    
    println!("Found {} images to analyze", image_paths.len());
    
    // Extract just the filenames for output
    let image_names: Vec<String> = image_paths
        .iter()
        .map(|path| path.file_name().unwrap_or_default().to_string_lossy().to_string())
        .collect();
    
    // Initialize the similarity matrix
    let num_images = image_paths.len();
    let mut similarity_matrix = vec![vec![0.0; num_images]; num_images];
    
    // Calculate similarities between all pairs of images
    let total_comparisons = num_images * num_images;
    let mut completed = 0;
    
    // Cache opened images to avoid reopening them
    let mut image_cache: HashMap<usize, image::DynamicImage> = HashMap::new();
    
    for i in 0..num_images {
        // Load image i if not already cached
        if !image_cache.contains_key(&i) {
            match image::open(&image_paths[i]) {
                Ok(img) => { image_cache.insert(i, img); },
                Err(e) => {
                    println!("Warning: Could not open image {}: {}", image_paths[i].display(), e);
                    continue;
                }
            }
        }
        
        // Compare with every other image (including itself)
        for j in 0..num_images {
            // Print progress
            completed += 1;
            if completed % 10 == 0 || completed == total_comparisons {
                print!("\rComparing images: {}/{} ({:.1}%)", 
                       completed, total_comparisons, 
                       (completed as f64 / total_comparisons as f64) * 100.0);
            }
            
            // Set diagonal to 1.0 (image is identical to itself)
            if i == j {
                similarity_matrix[i][j] = 1.0;
                continue;
            }
            
            // If we've already calculated j,i, use that value (symmetric matrix)
            if j < i {
                similarity_matrix[i][j] = similarity_matrix[j][i];
                continue;
            }
            
            // Load image j if not already cached
            if !image_cache.contains_key(&j) {
                match image::open(&image_paths[j]) {
                    Ok(img) => { image_cache.insert(j, img); },
                    Err(e) => {
                        println!("\nWarning: Could not open image {}: {}", image_paths[j].display(), e);
                        continue;
                    }
                }
            }
            
            // Calculate similarity
            let similarity = calculate_enhanced_similarity(
                &image_cache[&i], 
                &image_cache[&j], 
                config
            );
            
            // Store the result
            similarity_matrix[i][j] = similarity;
        }
    }
    
    println!("\nCompleted all image comparisons");
    
    let processing_time = start_time.elapsed();
    println!("Processing time: {:?}", processing_time);
    
    Ok(ConfusionMatrixResult {
        image_names,
        similarity_matrix,
        processing_time,
    })
}

/// Save confusion matrix to a CSV file
pub fn save_confusion_matrix_csv(
    result: &ConfusionMatrixResult,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    
    // Write header row
    write!(file, "Image")?;
    for name in &result.image_names {
        write!(file, ",{}", name)?;
    }
    writeln!(file)?;
    
    // Write data rows
    for i in 0..result.image_names.len() {
        write!(file, "{}", result.image_names[i])?;
        for j in 0..result.similarity_matrix[i].len() {
            write!(file, ",{:.4}", result.similarity_matrix[i][j])?;
        }
        writeln!(file)?;
    }
    
    println!("Confusion matrix saved to: {}", output_path.display());
    Ok(())
}

/// Save confusion matrix as a heatmap using Plotters (native Rust library)
pub fn save_confusion_matrix_plotters(
    result: &ConfusionMatrixResult,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    
    let n = result.image_names.len();
    if n == 0 {
        return Err("No images to plot".into());
    }
    
    // Determine appropriate dimensions
    let width = std::cmp::max(800, 100 + n * 40);
    let height = std::cmp::max(800, 100 + n * 40);
    
    // Create the plotting area
    let root = BitMapBackend::new(output_path, (width as u32, height as u32)).into_drawing_area();
    root.fill(&WHITE)?;
    
    // Calculate margins based on the longest filename
    let max_label_len = result.image_names.iter()
        .map(|name| name.len())
        .max()
        .unwrap_or(10);
    
    let margin = std::cmp::max(50, (max_label_len * 6) as i32);
    
    // Define the chart area
    let mut chart = ChartBuilder::on(&root)
        .caption("Image Similarity Matrix", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(margin)
        .y_label_area_size(margin)
        .build_cartesian_2d(0..n, (0..n).into_segmented())?;
    
    // Configure the mesh
    chart.configure_mesh()
        .disable_mesh()
        .x_labels(n)
        .y_labels(n)
        .x_label_formatter(&|x| {
            if *x < n {
                result.image_names[*x].clone()
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| {
            if let SegmentValue::Exact(y) = y {
                if *y < n {
                    result.image_names[*y].clone()
                } else {
                    String::new()
                }
            } else {
                String::new()
            }
        })
        .x_label_style(("sans-serif", 12).into_font().transform(FontTransform::Rotate90))
        .draw()?;
    
    // Create color gradient for heatmap
    let heatmap_gradient = colorous::VIRIDIS;
    
    // Draw the heatmap
    for i in 0..n {
        for j in 0..n {
            let value = result.similarity_matrix[i][j];
            
            // Map similarity value (0.0-1.0) to color
            let color = heatmap_gradient.eval_continuous(value as f64);
            let cell_color = RGBColor(color.r, color.g, color.b);
            
            // Draw the cell
            chart.draw_series(std::iter::once(
                Rectangle::new(
                    [(j, SegmentValue::Exact(i)), (j+1, SegmentValue::Exact(i+1))],
                    cell_color.filled()
                )
            ))?;
        }
    }
    
    // Draw a color scale
    let color_scale_area = root.margin(10, 10, 10, margin + 20).titled(
        "Similarity Scale",
        ("sans-serif", 15).into_font(),
    )?;
    
    let mut color_scale_chart = ChartBuilder::on(&color_scale_area)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(0)
        .build_cartesian_2d(0.0..1.0, 0..1)?;
    
    for i in 0..100 {
        let value = i as f64 / 100.0;
        let color = heatmap_gradient.eval_continuous(value);
        let cell_color = RGBColor(color.r, color.g, color.b);
        
        color_scale_chart.draw_series(std::iter::once(
            Rectangle::new(
                [(value, 0), (value + 0.01, 1)],
                cell_color.filled()
            )
        ))?;
    }
    
    color_scale_chart.configure_mesh()
        .x_labels(6)
        .y_labels(0)
        .draw()?;
    
    // Present the result
    root.present()?;
    
    println!("Plotters heatmap saved to: {}", output_path.display());
    Ok(())
}

/// Save confusion matrix as a heatmap HTML file
pub fn save_confusion_matrix_heatmap(
    result: &ConfusionMatrixResult,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(output_path)?;
    
    // Convert matrix to JSON
    let mut matrix_json = String::new();
    for (i, row) in result.similarity_matrix.iter().enumerate() {
        if i > 0 {
            matrix_json.push_str(",\n");
        }
        matrix_json.push_str("[");
        for (j, &val) in row.iter().enumerate() {
            if j > 0 {
                matrix_json.push_str(",");
            }
            matrix_json.push_str(&format!("{:.4}", val));
        }
        matrix_json.push_str("]");
    }
    
    // Create labels JSON
    let mut labels_json = String::new();
    for (i, name) in result.image_names.iter().enumerate() {
        if i > 0 {
            labels_json.push_str(",");
        }
        labels_json.push_str(&format!("\"{}\"", name));
    }
    
    // Write HTML with embedded JavaScript for heatmap
    write!(file, r#"<!DOCTYPE html>
<html>
<head>
    <title>Image Similarity Confusion Matrix</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #matrix {{ width: 100%; height: 800px; }}
        .info {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Image Similarity Confusion Matrix</h1>
    <div class="info">
        <p>Processing time: {:?}</p>
        <p>Number of images: {}</p>
        <p>Matching method: {}</p>
    </div>
    <div id="matrix"></div>
    <script>
        const data = [{{
            z: [
                {}
            ],
            x: [{}],
            y: [{}],
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            zmin: 0,
            zmax: 1
        }}];
        
        const layout = {{
            title: 'Image Similarity Matrix',
            annotations: [],
            xaxis: {{
                ticks: '',
                side: 'top',
                tickangle: -45
            }},
            yaxis: {{
                ticks: '',
                ticksuffix: ' '
            }},
            width: 1000,
            height: 1000
        }};
        
        Plotly.newPlot('matrix', data, layout);
    </script>
</body>
</html>
"#, result.processing_time, result.image_names.len(), 
       format!("{:?}", result.image_names), matrix_json, labels_json, labels_json)?;
    
    println!("Interactive heatmap saved to: {}", output_path.display());
    Ok(())
}

/// Run confusion matrix generation with the given directory
pub fn run_confusion_matrix(config: &Config, source_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Determine source and output paths
    let source_path = Path::new(source_dir);
    if !source_path.exists() || !source_path.is_dir() {
        return Err(format!("Source directory does not exist or is not a directory: {}", source_dir).into());
    }
    
    // Create output directory (based on source directory name)
    let dir_name = source_path.file_name()
        .unwrap_or_default()
        .to_string_lossy();
    
    let output_base_dir = config.get_output_dir().join("confusion_matrix").join(&*dir_name);
    fs::create_dir_all(&output_base_dir)?;
    
    // Generate the confusion matrix
    let result = generate_confusion_matrix(source_path, config)?;
    
    // Save the results
    let csv_path = output_base_dir.join("similarity_matrix.csv");
    save_confusion_matrix_csv(&result, &csv_path)?;
    
    // Save heatmap using Plotters (native Rust)
    let plotters_path = output_base_dir.join("similarity_heatmap.png");
    save_confusion_matrix_plotters(&result, &plotters_path)?;
    
    // Save interactive HTML heatmap
    let html_path = output_base_dir.join("similarity_heatmap.html");
    save_confusion_matrix_heatmap(&result, &html_path)?;
    
    println!("\nConfusion matrix analysis complete!");
    println!("Results saved to: {}", output_base_dir.display());
    println!("CSV data: {}", csv_path.display());
    println!("Static heatmap (Plotters): {}", plotters_path.display());
    println!("Interactive heatmap (HTML): {}", html_path.display());
    
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