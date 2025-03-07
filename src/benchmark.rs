use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

use crate::config::{Config, MatchingMethod};
use crate::categorize::{categorize_images, BenchmarkResult};

/// Run benchmark mode with all matching methods
pub fn run_benchmark(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    // Only proceed if benchmark mode is enabled
    if !config.benchmark_mode {
        println!("Benchmark mode is disabled in config. Skipping benchmark.");
        return Ok(());
    }
    
    println!("Running benchmark mode with all matching methods...");
    
    // Define all matching methods to benchmark
    let methods = vec![
        MatchingMethod::PerceptualHash,
        MatchingMethod::ColorHash,
        #[cfg(feature = "opencv")]
        MatchingMethod::Sift, 
        #[cfg(feature = "opencv")]
        MatchingMethod::Surf,
        MatchingMethod::Ssim,
    ];
    
    // Store results for each method
    let mut results: Vec<BenchmarkResult> = Vec::new();
    
    // Create benchmark base directory
    let benchmark_base_dir = Path::new(&config.output_directory).join("benchmark");
    fs::create_dir_all(&benchmark_base_dir)?;
    
    // Run categorization with each method
    for method in methods {
        // Clone the base config and customize for this run
        let mut method_config = config.clone();
        
        // Set the matching method for this run
        method_config.matching_method = method.clone();
        
        // Create a method-specific output directory
        let method_name = format!("{:?}", method).to_lowercase();
        method_config.output_directory = benchmark_base_dir
            .join(&method_name)
            .to_string_lossy()
            .to_string();
        
        println!("\n========================================");
        println!("Running benchmark with method: {:?}", method);
        println!("Output directory: {}", method_config.output_directory);
        println!("========================================\n");
        
        // Run categorization with this configuration
        match categorize_images(&method_config)? {
            Some(result) => {
                println!("Benchmark result for {:?}:\n{}", method, result.to_string());
                results.push(result);
            },
            None => {
                println!("No results for method {:?}", method);
            }
        }
    }
    
    // Generate benchmark report
    generate_benchmark_report(&benchmark_base_dir, &results)?;
    
    println!("\nBenchmark completed successfully. Report saved to {:?}", 
             benchmark_base_dir.join("benchmark_report.txt"));
    Ok(())
}

/// Generate a report of benchmark results
fn generate_benchmark_report(benchmark_dir: &Path, results: &[BenchmarkResult]) -> Result<(), Box<dyn std::error::Error>> {
    let report_path = benchmark_dir.join("benchmark_report.txt");
    let mut report = File::create(report_path)?;
    
    writeln!(report, "Media Lake Benchmark Report")?;
    writeln!(report, "==========================")?;
    writeln!(report, "")?;
    
    for result in results {
        writeln!(report, "{}", result.to_string())?;
    }
    
    // Add summary section
    writeln!(report, "\nSummary")?;
    writeln!(report, "-------")?;
    
    // Find the fastest method
    if let Some(fastest) = results.iter().min_by_key(|r| r.processing_time) {
        writeln!(report, "Fastest method: {} ({:?})", fastest.method_name, fastest.processing_time)?;
    }
    
    // Find the method that found the most categories
    if let Some(most_categories) = results.iter().max_by_key(|r| r.categories_count) {
        writeln!(report, "Method with most categories: {} ({} categories)", 
                 most_categories.method_name, most_categories.categories_count)?;
    }
    
    Ok(())
} 